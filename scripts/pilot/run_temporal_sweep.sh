#!/usr/bin/env bash
# Temporal-horizon sweep for the Gate-1 follow-up.
#
# Default: SHD T=25/50/100/200, seed 42, all four conditions.
# Generated YAMLs are saved under $RESULTS_DIR/generated_configs for auditability.
#
# Examples:
#   GPUS=0,1 RESULTS_DIR=pilot_results_t_sweep bash scripts/pilot/run_temporal_sweep.sh
#   TS="100 200" CONDITIONS="c1_lowrank c3_mga" SEEDS="42 123 2024" bash scripts/pilot/run_temporal_sweep.sh
#   DATA_ROOT=/data/maluzhang-folder/datasets/SHD BATCH_SIZE=96 bash scripts/pilot/run_temporal_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-shd}"                    # shd or dvs128
TS="${TS:-25 50 100 200}"
CONDITIONS="${CONDITIONS:-c0_sdla c1_lowrank c2_oneminusk c3_mga}"
SEEDS="${SEEDS:-42}"
GPUS="${GPUS:-0}"
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

make_configs() {
  local T="$1"; local cond="$2"; local out_dir="$RESULTS_DIR/generated_configs/${TASK}_T${T}"
  mkdir -p "$out_dir"
  DATA_CFG="$out_dir/data.yaml"
  TRAIN_CFG="$out_dir/train.yaml"
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
  local cmd="CUDA_VISIBLE_DEVICES=$gpu uv run python -m scripts.pilot.train_pilot \
    --task $task_label --condition $cond --seed $seed \
    --project_config configs/default_project_configs.yaml \
    --data_config $DATA_CFG \
    --train_config $TRAIN_CFG \
    --model_config $MODEL_CFG \
    --optimizer_config $OPT_CFG \
    --results_dir $RESULTS_DIR"
  echo "[T-sweep] gpu=$gpu task=$task_label cond=$cond seed=$seed -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  $cmd"
  else
    eval "$cmd" >"$log" 2>&1
  fi
}

i=0
declare -a PIDS=()
for T in $TS; do
  for cond in $CONDITIONS; do
    gpu="${GPU_ARR[$(( i % N_GPUS ))]}"
    i=$(( i + 1 ))
    (
      for seed in $SEEDS; do
        run_one "$T" "$cond" "$seed" "$gpu"
      done
    ) &
    PIDS+=($!)
  done
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "[T-sweep] one runner returned non-zero" >&2
done

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
