from __future__ import annotations

import os
import datetime
import functools
import yaml

import torch
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.load_config import load_config
from utils.resume import register_checkpoint_safe_globals, resolve_resume_ckpt
from utils.callbacks import EpochTimerCallback

from data import VisionDataModule
from models.build_model import build_model

def parser_args():
    import argparse
    parser = argparse.ArgumentParser(description="nanoSNN vision training")
    parser.add_argument("--project_config", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--optimizer_config", type=str, required=True)
    parser.add_argument("--resume", type=str, default="auto",
                        help="auto / none / /path/to/xxx.ckpt")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    return parser.parse_args()


def is_global_zero_env():
    return int(os.environ.get("RANK", "0")) == 0

def _namespace_to_dict(value):
    if isinstance(value, list):
        return [_namespace_to_dict(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _namespace_to_dict(val) for key, val in vars(value).items()}
    return value


import gc
from types import SimpleNamespace

def _bytes_to_gb(b):
    return b / 1e9

def profile_layer_memory(model, x, name="root"):
    """Run forward pass and report memory growth per submodule."""
    hooks = []
    memory_before = {}
    memory_after = {}
    
    def make_pre_hook(n):
        def hook(module, input):
            torch.cuda.synchronize()
            memory_before[n] = torch.cuda.memory_allocated()
        return hook
    
    def make_post_hook(n):
        def hook(module, input, output):
            torch.cuda.synchronize()
            memory_after[n] = torch.cuda.memory_allocated()
        return hook
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf only
            continue
        h1 = module.register_forward_pre_hook(make_pre_hook(name))
        h2 = module.register_forward_hook(make_post_hook(name))
        hooks.extend([h1, h2])
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(x)
    
    for h in hooks:
        h.remove()
    
    # Print sorted by memory growth
    rows = []
    for name in memory_before:
        if name in memory_after:
            delta = memory_after[name] - memory_before[name]
            rows.append((name, delta))
    rows.sort(key=lambda x: -x[1])
    
    print(f"\n{'Module':<60s} {'Mem Δ (MB)':>12s}")
    print("-" * 72)
    for name, delta in rows[:30]:
        print(f"{name:<60s} {delta/1e6:>12.1f}")
    
    print(f"\nTotal peak: {_bytes_to_gb(torch.cuda.max_memory_allocated()):.2f} GB")
    return out

def train(args):
    torch.backends.cudnn.benchmark = True
    
    rank_zero_info("########## nanoSNN training ##########")

    project_config = load_config(args.project_config)
    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    optimizer_config = load_config(args.optimizer_config)

    exp_name = (
        f"{model_config.name}"
        f"_T{getattr(model_config, 'T', 4)}"
        f"_data.{data_config.name}"
        f"_bsz.{train_config.batch_size_per_gpu}"
        f"_lr.{optimizer_config.lr}"
    )

    if getattr(train_config, "random_seed", -1) >= 0:
        seed_everything(train_config.random_seed, workers=True)

    timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    gpus_per_node = int(os.environ.get("GPU_PER_NODE", "1"))
    num_nodes = int(os.environ.get("N_NODE", "1"))
    world_size = gpus_per_node * num_nodes

    torch.backends.cudnn.benchmark = True
    precision = train_config.trainer.precision
    if precision in ("32", "32-true", "fp32"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # strategy
    trainer_strategy = train_config.trainer.strategy
    if isinstance(trainer_strategy, str) and trainer_strategy.lower() == "fsdp":
        from models.build_model import LitVisionSNN
        auto_wrap_policy = functools.partial(
            __import__("torch.distributed.fsdp.wrap", fromlist=["transformer_auto_wrap_policy"])
            .transformer_auto_wrap_policy,
            transformer_layer_cls={LitVisionSNN},
        )
        trainer_strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, use_orig_params=True)

    default_root_dir = getattr(project_config, "output_dir", "./exp/outputs")
    ckpt_dir = args.ckpt_dir or os.path.join(default_root_dir, "checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # save configs
    if is_global_zero_env():
        os.makedirs(default_root_dir, exist_ok=True)
        merged_config_path = os.path.join(default_root_dir, f"{exp_name}_{timestamp}_config.yaml")
        merged_config = {
            "project_config": _namespace_to_dict(project_config),
            "data_config": _namespace_to_dict(data_config),
            "train_config": _namespace_to_dict(train_config),
            "model_config": _namespace_to_dict(model_config),
            "optimizer_config": _namespace_to_dict(optimizer_config),
        }
        with open(merged_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged_config, f, sort_keys=False, allow_unicode=True)
        rank_zero_info(f"Saved merged config: {merged_config_path}")

    # logger
    if world_size == 1 and is_global_zero_env():
        wandb_dir = os.path.join(default_root_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        logger = WandbLogger(project=exp_name, name=timestamp, save_dir=wandb_dir, offline=False)
    else:
        logger = TensorBoardLogger(
            save_dir=os.path.join(default_root_dir, "tensorboard"), name=exp_name
        )

    save_every_n = int(getattr(train_config.trainer, "save_every_n_train_steps", 1000))
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EpochTimerCallback(),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:03d}-{step:07d}",
            save_top_k=-1,
            save_last=True,
            every_n_train_steps=save_every_n,
            save_on_train_epoch_end=True,
        ),
    ]

    trainer_kwargs = dict(
        strategy=trainer_strategy,
        precision=precision,
        devices=gpus_per_node,
        num_nodes=num_nodes,
        max_epochs=train_config.trainer.max_epochs,
        gradient_clip_val=train_config.trainer.gradient_clip_val,
        gradient_clip_algorithm=train_config.trainer.gradient_clip_algorithm,
        log_every_n_steps=train_config.trainer.log_every_n_steps,
        check_val_every_n_epoch=getattr(train_config.trainer, "check_val_every_n_epoch", 1),
        val_check_interval=getattr(train_config.trainer, "val_check_interval", 1.0),
        enable_checkpointing=train_config.trainer.enable_checkpointing,
        accumulate_grad_batches=getattr(train_config.trainer, "accumulate_grad_batches", 1),
        limit_train_batches=getattr(train_config, "limit_train_batches", None),
        logger=logger,
        callbacks=callbacks,
    )

    datamodule = VisionDataModule(data_config=data_config, train_config=train_config)
    lit_model = build_model(
        model_config=model_config,
        optimizer_config=optimizer_config,
        train_config=train_config,
        data_config=data_config,
    )
    rank_zero_info(lit_model.model)
    total_params = sum(p.numel() for p in lit_model.parameters())
    rank_zero_info(f"Total params: {total_params:,}")

    trainer = Trainer(**trainer_kwargs)
    register_checkpoint_safe_globals()
    ckpt_path = resolve_resume_ckpt(args.resume, ckpt_dir)
    if ckpt_path:
        rank_zero_info(f"Resuming from: {ckpt_path}")
    else:
        rank_zero_info("Training from scratch.")
        
    # # profiling
    # BS = 4  # small batch to isolate per-layer cost
    # x = torch.randn(BS, 3, 224, 224, device='cuda', dtype=torch.float32)
    # lit_model = lit_model.cuda() # 整个 Lightning module 
    # lit_model.model.train()
    # profile_layer_memory(lit_model, x)

    trainer.fit(lit_model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    args = parser_args()
    train(args)
