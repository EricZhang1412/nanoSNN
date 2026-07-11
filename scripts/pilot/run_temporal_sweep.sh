#!/usr/bin/env bash
# Temporal-horizon sweep for the Gate-1 follow-up.
#
# Default: SHD T=25/50/100/200, seed 42, all four conditions.
# Launches independent single-device training processes. Use JOBS_PER_GPU to
# allow multiple runs to share each listed device; this is not DDP.
# Generated YAMLs are saved under $RESULTS_DIR/generated_configs for auditability.
#
# Examples:
#   GPUS=0,1 RESULTS_DIR=pilot_results_t_sweep bash scripts/pilot/run_temporal_sweep.sh
#   GPUS=0,1 JOBS_PER_GPU=3 RESULTS_DIR=pilot_results_t_sweep bash scripts/pilot/run_temporal_sweep.sh
#   TS="100 200" CONDITIONS="c1_lowrank c3_mga" SEEDS="42 123 2024" bash scripts/pilot/run_temporal_sweep.sh
#   DATA_ROOT=/path/to/SHD BATCH_SIZE=96 bash scripts/pilot/run_temporal_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-shd}"                    # shd or dvs128
TS="${TS:-25 50 100 200}"
CONDITIONS="${CONDITIONS:-c0_sdla c1_lowrank c2_oneminusk c3_mga}"
SEEDS="${SEEDS:-42}"
GPUS="${GPUS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"       # independent single-device runs per listed GPU/NPU
RESULTS_DIR="${RESULTS_DIR:-pilot_results_t_sweep}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"            # optional train_config.trainer.max_epochs override
BATCH_SIZE="${BATCH_SIZE:-}"            # optional batch_size_per_gpu override
DATA_ROOT="${DATA_ROOT:-}"              # optional data_config.root override
export DATA_ROOT MAX_EPOCHS BATCH_SIZE

mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/generated_configs"

case "$TASK" in
  shd)
    BASE_DATA_CFG="configs/data_configs/shd_pilot.yaml"
    BASE_TRAIN_CFG="configs/train_configs/pilot_shd.yaml"
    OPT_CFG="configs/optimizer_configs/pilot_shd.yaml"
    MODEL_PREFIX="configs/model_configs/pilot/spikformer_shd"
    ;;
  dvs128)
    BASE_DATA_CFG="configs/data_configs/dvs128gesture_pilot.yaml"
    BASE_TRAIN_CFG="configs/train_configs/pilot_dvs.yaml"
    OPT_CFG="configs/optimizer_configs/pilot_dvs.yaml"
    MODEL_PREFIX="configs/model_configs/pilot/spikformer_dvs"
    ;;
  *)
    echo "Unsupported TASK=$TASK (expected shd or dvs128)" >&2
    exit 1
    ;;
esac

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"
if (( N_GPUS < 1 )); then
  echo "GPUS must contain at least one device id" >&2
  exit 1
fi
if ! [[ "$JOBS_PER_GPU" =~ ^[0-9]+$ ]] || (( JOBS_PER_GPU < 1 )); then
  echo "JOBS_PER_GPU must be a positive integer, got: $JOBS_PER_GPU" >&2
  exit 1
fi

make_configs() {
  local T="$1"; local cond="$2"; local out_dir="$RESULTS_DIR/generated_configs/${TASK}_T${T}"
  mkdir -p "$out_dir"
  DATA_CFG="$out_dir/data_${cond}.yaml"
  TRAIN_CFG="$out_dir/train_${cond}.yaml"
  MODEL_CFG="$out_dir/model_${cond}.yaml"
  python - "$BASE_DATA_CFG" "$BASE_TRAIN_CFG" "${MODEL_PREFIX}_${cond}.yaml" "$DATA_CFG" "$TRAIN_CFG" "$MODEL_CFG" "$T" <<'PY'
import os
import sys
import yaml

base_data, base_train, base_model, out_data, out_train, out_model, T = sys.argv[1:]
T = int(T)

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

data = load(base_data)
train = load(base_train)
model = load(base_model)

data["T"] = T
data["frames_number"] = T
if os.environ.get("DATA_ROOT"):
    data["root"] = os.environ["DATA_ROOT"]

model["T"] = T

if os.environ.get("BATCH_SIZE"):
    train["batch_size_per_gpu"] = int(os.environ["BATCH_SIZE"])
if os.environ.get("MAX_EPOCHS"):
    train.setdefault("trainer", {})["max_epochs"] = int(os.environ["MAX_EPOCHS"])

for obj, path in ((data, out_data), (train, out_train), (model, out_model)):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
PY
}

run_one() {
  local T="$1"; local cond="$2"; local seed="$3"; local gpu="$4"
  make_configs "$T" "$cond"
  local task_label="${TASK}_T${T}"
  local log="$RESULTS_DIR/logs/${task_label}_${cond}_seed${seed}.log"
  local -a cmd=(
    uv run python -m scripts.pilot.train_pilot
    --task "$task_label" --condition "$cond" --seed "$seed"
    --project_config configs/default_project_configs.yaml
    --data_config "$DATA_CFG"
    --train_config "$TRAIN_CFG"
    --model_config "$MODEL_CFG"
    --optimizer_config "$OPT_CFG"
    --results_dir "$RESULTS_DIR"
  )
  echo "[T-sweep] gpu=$gpu task=$task_label cond=$cond seed=$seed -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  CUDA_VISIBLE_DEVICES=$gpu ASCEND_RT_VISIBLE_DEVICES=$gpu DEVICES_PER_NODE=1 GPU_PER_NODE=1 NPU_PER_NODE=1 N_NODE=1 ${cmd[*]}"
  else
    CUDA_VISIBLE_DEVICES="$gpu" \
    ASCEND_RT_VISIBLE_DEVICES="$gpu" \
    DEVICES_PER_NODE=1 \
    GPU_PER_NODE=1 \
    NPU_PER_NODE=1 \
    N_NODE=1 \
      "${cmd[@]}" >"$log" 2>&1
  fi
}

declare -a JOB_TS=()
declare -a JOB_CONDS=()
for T in $TS; do
  for cond in $CONDITIONS; do
    JOB_TS+=("$T")
    JOB_CONDS+=("$cond")
  done
done

worker() {
  local worker_idx="$1"; local gpu="$2"; local total_slots="$3"; local status=0
  local job_idx T cond seed task_label
  for ((job_idx = worker_idx; job_idx < ${#JOB_TS[@]}; job_idx += total_slots)); do
    T="${JOB_TS[$job_idx]}"
    cond="${JOB_CONDS[$job_idx]}"
    task_label="${TASK}_T${T}"
    for seed in $SEEDS; do
      if ! run_one "$T" "$cond" "$seed" "$gpu"; then
        echo "[T-sweep] FAILED gpu=$gpu task=$task_label cond=$cond seed=$seed; see $RESULTS_DIR/logs/${task_label}_${cond}_seed${seed}.log" >&2
        status=1
      fi
    done
  done
  return "$status"
}

declare -a PIDS=()
declare -a SLOT_GPUS=()
for ((slot = 0; slot < JOBS_PER_GPU; slot += 1)); do
  for ((gpu_idx = 0; gpu_idx < N_GPUS; gpu_idx += 1)); do
    SLOT_GPUS+=("${GPU_ARR[$gpu_idx]}")
  done
done
TOTAL_SLOTS="${#SLOT_GPUS[@]}"
echo "[T-sweep] scheduling ${#JOB_TS[@]} task(s) across $N_GPUS device(s) x $JOBS_PER_GPU job(s)/device = $TOTAL_SLOTS single-device worker(s)"
for ((slot_idx = 0; slot_idx < TOTAL_SLOTS; slot_idx += 1)); do
  worker "$slot_idx" "${SLOT_GPUS[$slot_idx]}" "$TOTAL_SLOTS" &
  PIDS+=($!)
done

failures=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failures=$(( failures + 1 ))
  fi
done
if (( failures > 0 )); then
  echo "[T-sweep] $failures device worker(s) returned non-zero; skipping aggregation." >&2
  exit 1
fi

TASK_LABELS=""
for T in $TS; do TASK_LABELS+="${TASK}_T${T} "; done
echo "[T-sweep] aggregating: $TASK_LABELS"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[T-sweep] DRY_RUN=1, skipping aggregation"
else
  uv run python -m scripts.pilot.aggregate_results \
    --results_dir "$RESULTS_DIR" \
    --tasks "$TASK_LABELS" \
    --conditions "$CONDITIONS" \
    --seeds "$SEEDS"
fi
