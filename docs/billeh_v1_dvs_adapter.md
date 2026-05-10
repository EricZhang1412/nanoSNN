# Billeh-V1 + LGN 适配 DVS 数据集

> CIFAR10-DVS 和 DVS128Gesture 如何接入 V1 视觉皮层启发模型 (`billeh_v1`)

## 1. 背景与问题陈述

`billeh_v1` 是基于 Allen Institute V1 模型（Billeh et al., 2020）+ Chen-Maass 训练范式（Chen & Maass, 2022）实现的 SNN 视觉分类器。它的天然输入是 **LGN firing rate movie**：一个空间上覆盖 240×120 视网膜拓扑场、每个时间步给 17400 个 LGN cell 输出 Hz 单位 firing rate 的灰度视频。原始定义的处理时序按 **dt = 1 ms** 推进，V1 column 内部的 GLIF 神经元和 LGN 时域核都按这个粒度工作。

而 DVS 数据集（事件相机）默认输入：

| 数据集 | 原生 shape | 含义 |
|---|---|---|
| CIFAR10-DVS | `[T_dvs=16, 2, 128, 128]` | 16 个时间窗口积分帧 × ON/OFF 两个 polarity 通道 |
| DVS128Gesture | `[T_dvs=16, 2, 128, 128]` | 同上，但每段对应一个手势类别 |

直接喂进 `billeh_v1` 有 **三个根本不匹配**：

1. **Polarity 通道**：DVS 有 ON / OFF 两路，而 LGN 自然只接收一个标量亮度信号
2. **时间分辨率**：DVS 一段样本 16 帧 × ~62 ms/帧 = ~1 s，但 LGN 时域核 + V1 动力学是 ms 级
3. **空间映射**：DVS 128×128 ≠ LGN 240×120 retinotopic field

这份文档说明这三点是如何被解决的，以及最终 pipeline 的端到端数据流。

---

## 2. 三个适配点

### 2.1 Polarity：signed-diff 而非 mean

**问题**：DVS 像素的 ON 通道记录正向亮度变化（光增强）次数、OFF 通道记录负向亮度变化（光减弱）次数。直觉上最简单的合成方式是 `mean = (ON + OFF) / 2`，但这丢掉了正负号 —— "光在增强" 和 "光在减弱" 被混成同一个 "运动强度" 量。

**LGN 已经建模了 ON / OFF**：LGN 模块 (`models/billeh_v1/lgn_torch.py`) 里 17400 个 cell 中约 2/3 是单子区 cell（ON-only 或 OFF-only），另 1/3 是 ON-OFF 复合 cell：
- ON-cell 对**正向**亮度变化产生大的 firing rate 响应
- OFF-cell 对**负向**亮度变化产生大的 firing rate 响应

所以最自然的输入是一个 **signed** 亮度变化信号 —— 让 LGN 内置的 ON/OFF 通路自动激活。

**实现**：`data/transforms.py:build_dvs_lgn_transform`
```python
signed = tensor[:, 0] - tensor[:, 1]   # [T_dvs, H, W]，ON - OFF
```

然后做 per-sample 量纲归一化到 ~`[-1, 1]`：
```python
denom = signed.abs().amax().clamp_min(1e-3)
signed = signed / denom
```

这一步在 **数据侧**（dataset transform）做，而不是模型侧 —— polarity 是数据集属性，不该泄漏到 V1 模型里。

### 2.2 时间轴：K-replay 把 16 帧拉到 1024 LGN 步

**问题**：spikingjelly 主流惯例是把每段 DVS 录制集成成 `T_dvs ∈ {10, 16, 20}` 帧。直接用 T=16 喂 LGN 会有两个后果：
- LGN 时域核（典型 30-50 ms 支撑）在 16 步内只覆盖 1-2 帧，本质上**没用上时域核**
- V1 GLIF 神经元的膜电位/适应电流动力学需要 ms 级时间尺度才有意义，16 步过于粗糙
- rate-cost 正则（按神经元在整段时间的平均 firing rate 计算）只有 16 个采样点，方差极大

**方案**：每个 DVS 帧 hold K 个 LGN 时间步。
```
T_lgn = K × T_dvs = 64 × 16 = 1024
```
LGN 看到一个 16-段的"慢电影"，每段 64 ms。

**为什么 K=64**：
- 1024 跟 seq-CIFAR LGN 配置一致，复用 trial-timing（`pre_delay=64, down_sample=64` 给 16 个 readout chunks，跳第一段的 LGN 瞬态）
- 每帧 64 ms 给 LGN 时域核（最长 ~50 ms）和 V1 膜电位有足够的稳态时间

**实现**：`models/billeh_v1/model.py:_to_b_t_n_via_lgn` 的 ndim=5 分支
```python
t_in = movie.shape[1]            # 16
t_lgn = int(self.T)              # 1024
if t_lgn != t_in:
    if t_lgn % t_in != 0:
        raise ValueError(...)
    k = t_lgn // t_in            # 64
    movie = movie.repeat_interleave(k, dim=1)
```

注意 K-replay 在 **模型侧** 做，不在数据侧 —— `T_lgn` 是模型的属性（取决于 `model_config.T`），数据 transform 不应该知道它。

### 2.3 空间分辨率：让 LGN 自己处理

**LGN 怎么映射任意 H×W**：`lgn_torch.py:_spatial_response` 里 17400 个 cell 的 `(x_raw, y_raw)` 坐标定义在固定的 240×120 retinotopic field 里。`grid_sample` 用 bilinear 在任意尺寸的输入图像上采样：
```python
x_scaled = x * (w - 1) / 239.0
y_scaled = y * (h - 1) / 119.0
```
也就是说 LGN **不要求** 输入是 240×120，可以是 32×32（seq-CIFAR）也可以是 128×128（gesture），用 `grid_sample` 自动缩放。

**选择**：
- **CIFAR10-DVS**：原生 128×128 → bilinear resize 到 **48×48**（数据 transform 内完成）。理由：和主流 SNN baseline（Spikformer / QKFormer / SDT 等）横向可比，4090 显存压力小
- **DVS128Gesture**：保留**原生 128×128**。理由：手势数据本来就需要更大的空间细节，且 spikingjelly 的官方测试集就是 128×128

`lgn_input_height / lgn_input_width` 配置项告诉模型预期输入尺寸（仅用于 ndim=3 路径下的尺寸校验，ndim=5 路径不需要）。

---

## 3. 完整 pipeline 数据流

```
spikingjelly DVS dataset                  [T_dvs=16, 2, H_native=128, W_native=128]
    │     uint16 事件计数
    │
    │ ─── build_dvs_lgn_transform ───
    │     • signed = ON − OFF                                        [T_dvs, H, W]
    │     • bilinear resize 到 image_size                            [T_dvs, H_out, W_out]
    │     • per-sample 量纲归一化 |max| → 1                           ~[-1, 1]
    ▼
signed grayscale frames                   [T_dvs=16, 1, H_out, W_out]
    │
    │ ─── event_collate_fn ───
    │     batching + transpose (B 维插到第 1 位)
    ▼
batched event movie                       [T_dvs=16, B, 1, H_out, W_out]   (ndim=5)
    │
    │ ─── model._to_b_t_n_via_lgn ──
    │     • permute → [B, T_dvs, 1, H, W]
    │     • squeeze C (mean(dim=2) for C=1 is identity)
    │     • repeat_interleave(K=64, dim=1)                           [B, 1024, H, W]
    ▼
LGN-ready slow movie                      [B, T_lgn=1024, H, W]
    │
    │ ─── TorchLGN (冻结) ────────
    │     1) spatial: 各 cell 自己的 RF 大小做 2D Gaussian conv
    │        + bilinear sample at (x_raw, y_raw)
    │     2) temporal: 每个 cell 的 1D 因果 kernel (depthwise conv1d)
    │     3) ReLU(amp × filtered + spontaneous_rate)
    │     4) ON / OFF / 复合 cell 合成
    ▼
LGN firing rates (Hz)                     [B, 1024, 17400]
    │
    │ ─── × lgn_input_scale (9e-3) ──
    │     Hz → V1 input current 量级
    ▼
V1 input currents                         [B, 1024, 17400]
    │
    │ ─── BillehColumnTorch (V1) ──
    │     • LGN→V1 稀疏投射 (input_population)
    │     • 背景噪声 (bkg_weights)
    │     • 3000 个 GLIF 神经元 + Dale's law 约束的循环连接
    ▼
V1 spikes                                 [B, 1024, 3000]
    │
    │ ─── LocalizedPoolReadout ───
    │     L5e 神经元按 (x, y, z) 分 num_classes 个 30-neuron pool
    │     • CIFAR10-DVS (10 类): pool_offset=5, 用 pools 5..14 (legacy)
    │     • DVS128Gesture (11 类): pool_offset=0, 用 pools 0..10
    │     按 down_sample=64 ms 分 chunk 平均
    ▼
logits chunks → 收缩到 inference logits   [B, num_classes]
```

---

## 4. 数据集差异速查

| 维度 | CIFAR10-DVS | DVS128Gesture |
|---|---|---|
| 类别数 | 10 | 11 |
| Train / Test 划分 | 单 split 10k → 随机 9:1 切 | 官方按 user 切 1176 / 288 |
| 空间分辨率 (输入 V1 前) | 48×48 (resize) | 128×128 (原生) |
| 模型 `chunk_size` | 64 | 32 (内存压力大 → 减半) |
| `lgn_input_scale` | 9.0e-3 (calibrated) | 9.0e-3 (复用，初始 rate 22 Hz) |
| 数据可否自动下载 | ✅ spikingjelly 自动 | ❌ 需手动放 `DvsGesture.tar.gz` |
| Pool readout 路径 | offset=5 (legacy) | offset=0 (fixed for >10-class) |
| 初始 V1 population rate | ~23 Hz | ~22 Hz |

---

## 5. 关键配置文件

每个数据集需要 4 份 config（除了 train_config 跟 optimizer_config 是共用的）：

```
configs/data_configs/
    cifar10dvs_lgn.yaml        # 数据
    dvs128gesture_lgn.yaml

configs/model_configs/
    billeh_v1_cifar10dvs_lgn.yaml  # 模型
    billeh_v1_dvsgesture_lgn.yaml

configs/optimizer_configs/
    billeh_seqcifar10.yaml     # 共用：rate_cost=0.1, voltage_cost=1e-5

configs/train_configs/
    transformer.yaml           # 共用：bsz=4/GPU, bf16-mixed, max_epochs=100
```

### 数据 config 的关键字段

```yaml
transform_type: dvs_lgn       # 触发 build_dvs_lgn_transform 而不是默认 transform
polarity_mode: signed         # signed (推荐) | mean (ablation)
image_size: 48                # 或 128；transform 内 bilinear resize
frames_number: 16             # T_dvs，spikingjelly 集成的帧数
split_by: number              # spikingjelly 集成模式
```

### 模型 config 的关键字段

```yaml
T: 1024                       # T_lgn；必须能被 frames_number 整除（→ K=64）
use_lgn: true
auto_n_input_from_lgn: true   # n_input ← LGN cell count = 17400
auto_n_input_from_in_channels: false   # 关键：避免被 in_channels=1 覆盖
lgn_input_height: 48          # 或 128
lgn_input_width: 48
pre_delay: 64                 # 跳过第一段 LGN 瞬态
down_sample: 64               # readout chunk 大小（= K）
lgn_input_scale: 9.0e-3       # LGN Hz → V1 input current 的转换系数
```

### Optimizer config（共用 `billeh_seqcifar10.yaml`）

```yaml
rate_cost: 0.1                # Chen-Maass Huber-quantile rate loss 权重
voltage_cost: 1.0e-5
huber_kappa: 0.002
```

`rate_cost` 是把 V1 population rate **强制**约束在生物学水平的关键 —— 没有它，光靠 cross-entropy 训练 1-2 个 epoch 后 V1 就会饱和（rate 飙到 100+ Hz），loss 卡在 chance level。

---

## 6. 启动命令

### 单机多卡（4 GPUs，默认）

```bash
# CIFAR10-DVS
bash multigpu_train_cifar10dvs.sh

# DVS128Gesture
bash multigpu_train_dvsgesture.sh
```

### 单卡

```bash
source .venv/bin/activate

# CIFAR10-DVS
bash train.sh billeh_v1_cifar10dvs_lgn cifar10dvs_lgn billeh_seqcifar10

# DVS128Gesture
bash train.sh billeh_v1_dvsgesture_lgn dvs128gesture_lgn billeh_seqcifar10
```

### Smoke 验证（不训练，只跑一个 forward + backward 看初始 rate）

```bash
.venv/bin/python scripts/smoke_billeh_v1_dvs.py \
    --model_config configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml \
    --data_config configs/data_configs/cifar10dvs_lgn.yaml \
    --batch 2
```

期望输出（关键三行）：
```
[smoke] model T=1024, n_input=17400, num_classes=10
[smoke] population mean rate ≈ 23.93 Hz       ← 落在 [5, 30] 区间
[smoke] OK
```

---

## 7. `lgn_input_scale` 校准方法

`lgn_input_scale` 把 LGN 输出的 Hz 缩放到 V1 input current 单位。这个系数取决于：
- LGN 输出能量（受输入信号 dynamic range 影响 —— DVS signed-diff、归一化方式、空间大小都有影响）
- V1 input weights 的量级（由 Allen Institute 标定，不变）

**当 V1 初始 rate 不在 [5, 30] Hz 时，调这个系数**。校准方法（迭代二分）：

| 实测 rate | 怎么调 `lgn_input_scale` |
|---|---|
| 远低于目标（< 1 Hz） | × 10 |
| 略低 | × 3 |
| 落在 [5, 30] Hz | 停 |
| 略高（30-60 Hz） | × 0.5 |
| 饱和（> 100 Hz） | × 0.2 |

rate 对 scale 是 **超线性**（V1 神经元的 spike threshold 导致），通常 2-3 次迭代能定下来。CIFAR10-DVS 的校准记录在 commit `dc17289`：1e-3 (0.12 Hz) → 5e-2 (131 Hz) → 1.5e-2 (45 Hz) → 9e-3 (23.93 Hz) ✓。

---

## 8. 11 类的 readout 修复

`models/billeh_v1/model.py` 里 readout pool 选择原本的逻辑：
```python
key = f"localized_readout_neuron_ids_{i + 5}"   # 默认偏移 5
if key not in network:
    key = f"localized_readout_neuron_ids_{i}"   # fallback
```

这是 Chen-Maass paper 留下来的 10-class image task 约定（pools 0..4 留给 garrett 2-class，pools 5..14 给 10-class image）。对于 `num_classes > 10` 的任务（如 DVS128Gesture 的 11 类），fallback 路径会让多个 class **共用同一个 pool**（class 5 和 class 10 都指向 pool 10），产生 readout 别名 bug。

修复后的逻辑：
```python
if self.num_classes == 10 and "localized_readout_neuron_ids_5" in network:
    pool_offset = 5    # legacy 10-class 路径
elif self.num_classes <= 15:
    pool_offset = 0    # 用前 num_classes 个 pool，无别名
else:
    raise ValueError("num_classes > 15 not supported by localized readout")
```

CIFAR10-DVS 走 offset=5（保持 legacy），DVS128Gesture 走 offset=0 用 pools 0..10。

---

## 9. 常见问题

### Q1: 为什么 `n_input` 必须是 17400，不能是 1？
`n_input` 决定 V1 input population 的输入维度。如果设成 RGB/polarity 通道数（1 或 2 或 3），底层 `reduce_input_population` 会把 17400 个 LGN-to-V1 的连接**求和**到这 1-3 个超级 channel 里 —— 等效权重被放大 ~5800 倍，V1 在初始化时就会饱和（rate > 200 Hz）。

设置 `auto_n_input_from_in_channels: false` + `auto_n_input_from_lgn: true` 保证 `n_input = lgn.n_cells = 17400`。

### Q2: 为什么不直接用 spikingjelly T_dvs=600 跳过 K-replay？
理论上可以让 spikingjelly 把事件细分到 600 帧（每帧 ~3 ms），但：
- 偏离主流 SNN baseline（普遍用 T=10-20）
- 每帧事件数太少，signed-diff 噪声极大
- 数据预处理时间也几倍延长

K-replay 把 spikingjelly 惯例和 LGN/V1 ms-resolution 解耦干净 —— 数据侧按主流做，模型侧自己拉时间。

### Q3: 为什么 polarity 用 signed-diff 不用 2 通道单独喂？
理论上 LGN 内部的 ON 通路和 OFF 通路可以分别接收两个通道，但当前 `TorchLGN.forward` 只接受单通道 movie。要做 2 通道需要把 `_spatial_response` 改成 dom / non_dom 分别取不同 movie —— 改动较大且生物学上不严格更优（real LGN 接收的是一个信号，ON/OFF 是细胞自身的响应特性）。signed-diff 用单通道复用现有 LGN 路径，最干净。

### Q4: DVS128Gesture 训练 OOM 怎么办？
128×128 在 4090 (24 GB) 上 bsz=4/GPU 可能 OOM。三种应对：
1. 改 `configs/train_configs/transformer.yaml` 把 `batch_size_per_gpu` 改成 2（影响其他训练）
2. 新建一份 `billeh_dvs128.yaml` train config，单独给 gesture 用 bsz=2 + `accumulate_grad_batches=2`
3. 把模型 config 的 `chunk_size: 32` 再降到 16（更细粒度的 gradient checkpointing，省显存但慢一些）

### Q5: spikingjelly 装好了但数据预处理报 `np.fromstring` 错？
numpy 2.0+ 移除了 `np.fromstring`，spikingjelly 0.0.0.0.14 还在用它。`venv/lib/.../spikingjelly/datasets/cifar10_dvs.py` 需要这两处 patch：
```python
np.fromstring(data, dtype='>u4')   →   np.frombuffer(data, dtype='>u4')   # line 62
.astype(np.bool)                    →   .astype(bool)                      # line 86
```
DVS128Gesture 走 `load_aedat_v3` 路径不受影响。

---

## 10. 涉及到的文件清单

**核心代码改动**：
- `data/transforms.py:build_dvs_lgn_transform` — polarity collapse + resize + 归一化
- `data/event_datasets.py:build_event_dataset` — `transform_type: dvs_lgn` dispatch 分支
- `models/billeh_v1/model.py:_to_b_t_n_via_lgn` — ndim=5 分支加 K-replay
- `models/billeh_v1/model.py` 构造函数 — 11-class readout pool fix

**新增 config**：
- `configs/data_configs/cifar10dvs_lgn.yaml`
- `configs/data_configs/dvs128gesture_lgn.yaml`
- `configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml`
- `configs/model_configs/billeh_v1_dvsgesture_lgn.yaml`

**测试 / 工具脚本**：
- `scripts/test_dvs_lgn_transform.py` — transform unit test
- `scripts/test_billeh_lgn_replay.py` — K-replay unit test
- `scripts/test_billeh_readout_pools.py` — 11-class disjoint pool test
- `scripts/smoke_billeh_v1_dvs.py` — 端到端 smoke

**训练入口**：
- `multigpu_train_cifar10dvs.sh`
- `multigpu_train_dvsgesture.sh`

**设计 / 实施文档**：
- `docs/superpowers/specs/2026-05-11-billeh-dvs-design.md` — 设计 spec
- `docs/superpowers/plans/2026-05-11-billeh-dvs.md` — 10 步 TDD 实施 plan

---

## 11. 参考

- Billeh, Y. N., Cai, B., et al. (2020). *Systematic Integration of Structural and Functional Data into Multi-Scale Models of Mouse Primary Visual Cortex.* Neuron 106, 388-403.
- Chen, G. & Maass, W. (2022). *A Computational Model of the Specific and Generic Mechanisms of Direction Selectivity and Speed Tuning in Mouse Visual Cortex V1.* Science Advances 8, eabq7592.
- CIFAR10-DVS: Li, H., et al. (2017). *CIFAR10-DVS: An Event-Stream Dataset for Object Classification.* Frontiers in Neuroscience 11, 309.
- DVS128Gesture: Amir, A., et al. (2017). *A Low Power, Fully Event-Based Gesture Recognition System.* CVPR.
- spikingjelly: Fang, W., et al. (2023). *SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence.* Science Advances 9, eadi1480.
