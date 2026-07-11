from __future__ import annotations

import os
import datetime
import functools

import torch
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.strategies import FSDPStrategy, SingleDeviceStrategy
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.ascend import NPUAccelerator, make_npu_ddp_strategy, resolve_accelerator
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


def _env_int(*names: str, default: int = 1) -> int:
    for name in names:
        value = os.environ.get(name)
        if value:
            return int(value)
    return int(default)


def _configure_torch_backend(accelerator_name: str, precision: str) -> None:
    if accelerator_name != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if precision in ("32", "32-true", "fp32"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True


def _build_logger(project_config, exp_name: str, timestamp: str):
    default_root_dir = getattr(project_config, "output_dir", "./exp/outputs")
    logger_name = str(os.environ.get("NANOSNN_LOGGER", "") or "tensorboard").lower()
    logger_name = str(getattr(project_config, "logger", logger_name) or logger_name).lower()

    if logger_name in {"none", "false", "off"}:
        return False

    if logger_name == "wandb":
        wandb_dir = os.path.join(default_root_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        offline = str(os.environ.get("WANDB_MODE", "")).lower() == "offline"
        return WandbLogger(project=exp_name, name=timestamp, save_dir=wandb_dir, offline=offline)

    return TensorBoardLogger(
        save_dir=os.path.join(default_root_dir, "tensorboard"), name=exp_name
    )


def _resolve_trainer_device_args(train_config, accelerator_name: str, devices_per_node: int, num_nodes: int):
    requested_strategy = train_config.trainer.strategy

    if accelerator_name != "npu":
        return accelerator_name, requested_strategy, devices_per_node, []

    npu_accelerator = NPUAccelerator()
    device_ids = type(npu_accelerator).parse_devices(devices_per_node)
    parallel_devices = type(npu_accelerator).get_parallel_devices(device_ids)
    strategy_name = str(requested_strategy).lower() if isinstance(requested_strategy, str) else requested_strategy

    if strategy_name == "fsdp":
        raise ValueError("FSDP is not supported by this Ascend NPU adapter; use strategy=auto or ddp.")
    if len(parallel_devices) == 1 and num_nodes == 1 and strategy_name in {"auto", "single_device"}:
        strategy = SingleDeviceStrategy(device=parallel_devices[0], accelerator=npu_accelerator)
    elif strategy_name in {"auto", "ddp"}:
        strategy = make_npu_ddp_strategy(parallel_devices, npu_accelerator)
    else:
        strategy = requested_strategy

    return "auto", strategy, device_ids, parallel_devices


def train(args):
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
    num_nodes = int(os.environ.get("N_NODE", "1"))
    requested_accelerator = str(
        os.environ.get(
            "NANOSNN_ACCELERATOR",
            getattr(train_config.trainer, "accelerator", "auto"),
        )
    ).lower()
    accelerator_name = resolve_accelerator(requested_accelerator)
    os.environ["NANOSNN_RESOLVED_ACCELERATOR"] = accelerator_name

    precision = train_config.trainer.precision
    devices_per_node = _env_int(
        "DEVICES_PER_NODE",
        "NPU_PER_NODE" if accelerator_name == "npu" else "GPU_PER_NODE",
        "GPU_PER_NODE",
        default=getattr(train_config.trainer, "devices", 1),
    )
    world_size = devices_per_node * num_nodes
    _configure_torch_backend(accelerator_name, precision)
    rank_zero_info(
        f"accelerator={accelerator_name}, devices_per_node={devices_per_node}, "
        f"num_nodes={num_nodes}, precision={precision}"
    )

    precision_plugin = None
    if accelerator_name == "npu" and precision in {"16-mixed", "bf16-mixed"}:
        precision_plugin = MixedPrecision(precision, "npu")
        precision_arg = None
    else:
        precision_arg = precision

    # strategy
    trainer_accelerator, trainer_strategy, trainer_devices, _ = _resolve_trainer_device_args(
        train_config, accelerator_name, devices_per_node, num_nodes
    )
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

    # logger
    logger = _build_logger(project_config, exp_name, timestamp)

    save_every_n = int(getattr(train_config.trainer, "save_every_n_train_steps", 1000))
    callbacks = [EpochTimerCallback()]
    if logger:
        callbacks.insert(0, LearningRateMonitor(logging_interval="step"))

    periodic_checkpoint_callback = None
    best_checkpoint_callback = None
    enable_checkpointing = bool(train_config.trainer.enable_checkpointing)
    if enable_checkpointing:
        periodic_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:03d}-{step:07d}",
            save_top_k=-1,
            save_last=True,
            every_n_train_steps=save_every_n,
            save_on_train_epoch_end=True,
        )
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:03d}-{step:07d}",
            monitor="val/top1",
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=False,
            every_n_epochs=1,
        )
        callbacks.extend([periodic_checkpoint_callback, best_checkpoint_callback])

    trainer_kwargs = dict(
        accelerator=trainer_accelerator,
        strategy=trainer_strategy,
        devices=trainer_devices,
        num_nodes=num_nodes,
        max_epochs=train_config.trainer.max_epochs,
        gradient_clip_val=train_config.trainer.gradient_clip_val,
        gradient_clip_algorithm=train_config.trainer.gradient_clip_algorithm,
        log_every_n_steps=train_config.trainer.log_every_n_steps,
        check_val_every_n_epoch=getattr(train_config.trainer, "check_val_every_n_epoch", 1),
        val_check_interval=getattr(train_config.trainer, "val_check_interval", 1.0),
        enable_checkpointing=enable_checkpointing,
        accumulate_grad_batches=getattr(train_config.trainer, "accumulate_grad_batches", 1),
        limit_train_batches=getattr(train_config, "limit_train_batches", None),
        limit_val_batches=getattr(train_config, "limit_val_batches", None),
        limit_test_batches=getattr(train_config, "limit_test_batches", None),
        logger=logger,
        callbacks=callbacks,
    )
    for optional_key in ("max_steps", "num_sanity_val_steps"):
        optional_value = getattr(train_config.trainer, optional_key, None)
        if optional_value is not None:
            trainer_kwargs[optional_key] = optional_value
    if precision_arg is not None:
        trainer_kwargs["precision"] = precision_arg
    if precision_plugin is not None:
        trainer_kwargs["plugins"] = [precision_plugin]

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

    trainer.fit(lit_model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_ckpt_path = None
    if best_checkpoint_callback is not None and best_checkpoint_callback.best_model_path:
        test_ckpt_path = best_checkpoint_callback.best_model_path
    elif periodic_checkpoint_callback is not None and periodic_checkpoint_callback.last_model_path:
        test_ckpt_path = periodic_checkpoint_callback.last_model_path
    trainer.test(lit_model, datamodule=datamodule, ckpt_path=test_ckpt_path)


if __name__ == "__main__":
    args = parser_args()
    train(args)
