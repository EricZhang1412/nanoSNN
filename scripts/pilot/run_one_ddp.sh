#!/usr/bin/env bash
# Run a single pilot (task, condition, seed) on N GPUs via torchrun DDP.
#
# IMPORTANT — batch-size caveat (pilot §2 isolation):
#   * Single-GPU baseline uses batch_size_per_gpu=16 (DVS) / 128 (SHD).
#   * With N GPUs and the YAML unchanged, *effective* batch = N×16 (or N×128).
#     That changes SGD noise level → optimization trajectory differs from the
#     other 3 conditions (which were trained single-GPU).
#   * For a fair comparison, either retrain ALL conditions with the same N,
#     or set `--scale_batch=1` to scale batch_size_per_gpu down by N so the
#     effective batch matches the single-GPU baseline.
#
# Usage:
#   GPUS=0,1,2,3 TASK=dvs128 COND=c3_mga SEED=42 [SCALE_BATCH=1] \
#     bash scripts/pilot/run_one_ddp.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0}"
TASK="${TASK:-dvs128}"
COND="${COND:-c3_mga}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:-pilot_results}"
PORT="${PORT:-29503}"
SCALE_BATCH="${SCALE_BATCH:-0}"   # 1 = scale batch_size_per_gpu down by N to preserve effective batch

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"

case "$TASK" in
  dvs128)
    DATA_CFG="configs/data_configs/dvs128gesture_pilot.yaml"
    TRAIN_CFG="configs/train_configs/pilot_dvs.yaml"
    OPT_CFG="configs/optimizer_configs/pilot_dvs.yaml"
    MODEL_CFG="configs/model_configs/pilot/spikformer_dvs_${COND}.yaml"
    SINGLE_GPU_BSZ=16
    ;;
  shd)
    DATA_CFG="configs/data_configs/shd_pilot.yaml"
    TRAIN_CFG="configs/train_configs/pilot_shd.yaml"
    OPT_CFG="configs/optimizer_configs/pilot_shd.yaml"
    MODEL_CFG="configs/model_configs/pilot/spikformer_shd_${COND}.yaml"
    SINGLE_GPU_BSZ=128
    ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

# Optionally rewrite the train_config to scale batch_size_per_gpu so the
# *effective* batch (sum across GPUs) matches the single-GPU baseline.
EFFECTIVE_TRAIN_CFG="$TRAIN_CFG"
if [[ "$SCALE_BATCH" == "1" ]]; then
  SCALED=$(( SINGLE_GPU_BSZ / N_GPUS ))
  if [[ "$SCALED" -lt 1 ]]; then
    echo "[ddp] N_GPUS=$N_GPUS > SINGLE_GPU_BSZ=$SINGLE_GPU_BSZ, can't scale" >&2
    exit 1
  fi
  TMP_CFG=$(mktemp --suffix=.yaml)
  python - <<PY > "$TMP_CFG"
import yaml
d = yaml.safe_load(open("$TRAIN_CFG"))
d["batch_size_per_gpu"] = $SCALED
print(yaml.safe_dump(d, sort_keys=False), end="")
PY
  EFFECTIVE_TRAIN_CFG="$TMP_CFG"
  echo "[ddp] SCALE_BATCH=1: batch_size_per_gpu = $SINGLE_GPU_BSZ → $SCALED (effective batch stays $SINGLE_GPU_BSZ)"
else
  EFFECTIVE=$(( SINGLE_GPU_BSZ * N_GPUS ))
  echo "[ddp] SCALE_BATCH=0: effective batch = $EFFECTIVE (single-GPU baseline was $SINGLE_GPU_BSZ)"
  echo "[ddp]   ⚠ this changes SGD noise vs. C0/C1/C2 → strict isolation is broken"
fi

CKPT_DIR="$RESULTS_DIR/checkpoints/$TASK/$COND/seed${SEED}"
JSON_DIR="$RESULTS_DIR/$TASK/$COND"
mkdir -p "$CKPT_DIR" "$JSON_DIR" "$RESULTS_DIR/logs"
LOG="$RESULTS_DIR/logs/${TASK}_${COND}_seed${SEED}_ddp${N_GPUS}.log"

export CUDA_VISIBLE_DEVICES="$GPUS"
export GPU_PER_NODE="$N_GPUS"
export N_NODE=1

echo "[ddp] task=$TASK cond=$COND seed=$SEED n_gpus=$N_GPUS  log=$LOG"

uv run torchrun \
  --nproc_per_node="$N_GPUS" \
  --nnodes=1 \
  --master_port="$PORT" \
  -m scripts.pilot.train_pilot \
    --task "$TASK" --condition "$COND" --seed "$SEED" \
    --project_config configs/default_project_configs.yaml \
    --data_config    "$DATA_CFG" \
    --train_config   "$EFFECTIVE_TRAIN_CFG" \
    --model_config   "$MODEL_CFG" \
    --optimizer_config "$OPT_CFG" \
    --results_dir    "$RESULTS_DIR" \
  2>&1 | tee "$LOG"
