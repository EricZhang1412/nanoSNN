# Ascend CANN 9.0 适配说明

本文档记录 nanoSNN 在 Ascend NPU 环境上的适配内容、环境要求、单卡/两卡运行命令和常见问题。当前适配已在 2 张 Ascend 910B3 NPU 上验证过 HCCL DDP、Lightning DDP 以及最小训练闭环。

## 1. 环境要求

推荐版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | `>=3.10,<3.12`，推荐 Python 3.11 |
| CANN | 9.0.0 |
| PyTorch | `torch==2.6.0` CPU wheel |
| torch-npu | `torch-npu==2.6.0` |
| torchvision | `torchvision==0.21.0` |
| triton-ascend | `3.2.0.dev20260515` |
| Lightning | 2.x |

每次安装依赖或运行训练前，先加载 Ascend 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果 CANN 安装在其他路径，请替换为实际的 `set_env.sh`。

安装依赖：

```bash
uv sync --python python3.11
```

## 2. 适配内容概览

本次适配主要覆盖以下部分：

- `utils/ascend.py`：新增 Lightning NPU Accelerator 与 HCCL DDP Strategy。
- `train.py`：支持 `NANOSNN_ACCELERATOR=npu`、`DEVICES_PER_NODE` / `NPU_PER_NODE`、NPU mixed precision plugin，以及 NPU DDP 初始化。
- `multigpu_train.sh`：改为通用多设备 `torchrun` 启动脚本，支持 Ascend NPU 两卡训练。
- `data/datamodule.py`：增加 `prepare_data()`，避免 DDP 多进程同时下载数据；NPU 下关闭 CUDA 专用 `pin_memory`。
- `models/build_model.py`：修复 DDP 指标同步；避免 Billeh 已经是 `[T, B, C, H, W]` 的输入被二次 temporal expand。
- `scripts/smoke_ascend_npu.py`：单卡 Ascend runtime / torch-npu / triton-ascend / nanoSNN forward-backward smoke。
- `scripts/smoke_hccl_ddp_npu.py`：HCCL all-reduce、原生 DDP、nanoSNN DDP step smoke。
- `scripts/smoke_lightning_hccl_npu.py`：Lightning + 自定义 NPU HCCL DDP strategy smoke。

## 3. 快速检查 NPU

```bash
npu-smi info

.venv/bin/python - <<'PY'
import torch
import torch_npu
print("npu_available", torch.npu.is_available())
print("npu_count", torch.npu.device_count())
PY
```

期望至少看到：

```text
npu_available True
npu_count 2
```

## 4. 单卡 smoke test

验证 CANN runtime、`torch_npu`、`triton-ascend` 以及一个 nanoSNN forward/backward step：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
uv run python -m scripts.smoke_ascend_npu
```

如果只想跳过 Triton 或模型 step：

```bash
uv run python -m scripts.smoke_ascend_npu --skip_triton
uv run python -m scripts.smoke_ascend_npu --skip_model
```

## 5. 两卡 HCCL / DDP smoke test

### 5.1 原生 HCCL + DDP + nanoSNN step

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1 \
.venv/bin/torchrun --nproc_per_node=2 --master_port=29513 \
  -m scripts.smoke_hccl_ddp_npu --require_multi
```

该脚本会验证：

- HCCL process group 初始化。
- `dist.all_reduce` 结果是否正确。
- 一个 tiny DDP 模型的 forward/backward/step。
- 一个 nanoSNN 模型的 DDP forward/backward/step。

### 5.2 Lightning HCCL DDP strategy

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1 \
.venv/bin/torchrun --nproc_per_node=2 --master_port=29514 \
  -m scripts.smoke_lightning_hccl_npu --require_multi
```

期望看到：

```text
distributed_backend=hccl
[PASS] Lightning HCCL/DDP smoke completed
```

## 6. 两卡训练命令

### 6.1 通用两卡命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1 \
NANOSNN_ACCELERATOR=npu \
bash multigpu_train.sh \
  default_project_configs \
  default \
  sdt_v1_small \
  cifar10 \
  sdtv1_cifar10 \
  2 \
  1 \
  29502
```

参数顺序：

```text
bash multigpu_train.sh [project] [train] [model] [data] [optimizer] [devices_per_node] [num_nodes] [port]
```

### 6.2 Billeh MNIST LGN 两卡命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1 \
NANOSNN_ACCELERATOR=npu \
bash multigpu_train.sh \
  default_project_configs \
  billeh_mnist \
  billeh_v1_mnist_lgn \
  billeh_mnist_lgn \
  billeh_mnist \
  2 \
  1 \
  29503
```

Billeh 相关数据默认路径在配置中指向：

```text
/path/to/GLIF_network2
```

需要确认该目录下至少包含：

- `network_dat.pkl`
- `network/v1_nodes.h5`
- `network/v1_node_types.csv`
- `garrett_firing_rates.pkl`
- `lgn_full_col_cells_3.csv`
- `temporal_kernels.pkl`

## 7. 关键环境变量

| 变量 | 说明 |
| --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES=0,1` | 指定可见 NPU 卡号。 |
| `NANOSNN_ACCELERATOR=npu` | 强制使用 Ascend NPU；不设置时 `auto` 会优先检测 NPU。 |
| `DEVICES_PER_NODE=2` | 每个节点的设备数，`multigpu_train.sh` 会自动设置。 |
| `NPU_PER_NODE=2` | NPU 设备数，兼容训练脚本读取。 |
| `N_NODE=1` | 节点数，单机两卡为 1。 |
| `TORCHRUN_BIN=/path/to/torchrun` | 可选，指定 torchrun；默认优先使用 `.venv/bin/torchrun`。 |
| `NANOSNN_LOGGER=none` | 可选，关闭 logger，适合 smoke。 |
| `WANDB_MODE=offline` | 可选，WandB 离线模式。 |

## 8. 已验证命令

在当前两卡 NPU 环境中已跑通：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 \
  --master_port=29613 scripts/smoke_hccl_ddp_npu.py --require_multi

ASCEND_RT_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 \
  --master_port=29611 scripts/smoke_lightning_hccl_npu.py --require_multi
```

也已用临时小配置验证 `train.py` 的两卡 `Trainer.fit()` + `Trainer.test()` 能正常结束，包括 SDT 与 Billeh 小配置。

## 9. 常见问题

### 9.1 `torch.npu.is_available()` 是 False

检查 CANN 环境是否已加载：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

如果仍为 False，检查 `torch` 与 `torch-npu` 版本是否匹配。

### 9.2 HCCL 初始化卡住或失败

建议先确认没有残留训练进程：

```bash
npu-smi info
```

然后换一个未占用端口：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 \
  --master_port=29601 -m scripts.smoke_lightning_hccl_npu --require_multi
```

### 9.3 Billeh 数据文件缺失

如果报 `billeh_data_dir not found` 或 `garrett_firing_rates.pkl not found`，请检查 `configs/model_configs/billeh_v1_*.yaml` 中的 `billeh_data_dir` 是否指向实际数据目录。

如果缺 `temporal_kernels.pkl`，可运行：

```bash
uv run python -m scripts.prepare_lgn_kernels \
  --lgn_data_path /path/to/GLIF_network2/lgn_full_col_cells_3.csv \
  --output /path/to/GLIF_network2/temporal_kernels.pkl
```

### 9.4 NPU 上 `pin_memory` 警告或无效

`pin_memory` 是 CUDA host memory 优化，本适配在 NPU 下会自动关闭，CUDA 下仍按 data config 生效。

## 10. 推理和开发建议

- 新增模型时，如果模型不需要 `[T, B, C, H, W]` temporal expansion，可设置 `expects_temporal_input=False`。
- DDP 下训练集、验证集和测试集均交给 Lightning 自动注入 `DistributedSampler`。
- 多卡训练前优先跑 `scripts.smoke_lightning_hccl_npu.py`，排除 HCCL 或环境问题后再跑完整训练。
