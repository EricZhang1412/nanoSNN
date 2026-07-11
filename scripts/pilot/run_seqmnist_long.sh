#!/usr/bin/env bash
# Long-horizon external benchmark: Sequential MNIST / Permuted Sequential MNIST.
#
# Examples:
#   GPUS=0,1 RESULTS_DIR=pilot_results_seqmnist bash scripts/pilot/run_seqmnist_long.sh
#   PERMUTE=1 SEEDS="42 123 2024" DATA_ROOT=/path/to/MNIST bash scripts/pilot/run_seqmnist_long.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0}"
RESULTS_DIR="${RESULTS_DIR:-pilot_results_seqmnist}"
CONDITIONS="${CONDITIONS:-c0_sdla c1_lowrank c2_oneminusk c3_mga}"
SEEDS="${SEEDS:-42}"
PERMUTE="${PERMUTE:-0}"
DATA_ROOT="${DATA_ROOT:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
DRY_RUN="${DRY_RUN:-0}"
export DATA_ROOT MAX_EPOCHS BATCH_SIZE PERMUTE

BASE_DATA_CFG="configs/data_configs/seqmnist_pilot.yaml"
BASE_TRAIN_CFG="configs/train_configs/pilot_seqmnist.yaml"
OPT_CFG="configs/optimizer_configs/pilot_seqmnist.yaml"
MODEL_PREFIX="configs/model_configs/pilot/spikformer_seqmnist"
TASK_LABEL="seqmnist"
if [[ "$PERMUTE" == "1" ]]; then
  TASK_LABEL="pseqmnist"
fi

mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/generated_configs"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"

make_configs() {
  local cond="$1"; local out_dir="$RESULTS_DIR/generated_configs/$TASK_LABEL"
  mkdir -p "$out_dir"
  DATA_CFG="$out_dir/data.yaml"
  TRAIN_CFG="$out_dir/train.yaml"
  MODEL_CFG="$out_dir/model_${cond}.yaml"
  python - "$BASE_DATA_CFG" "$BASE_TRAIN_CFG" "${MODEL_PREFIX}_${cond}.yaml" "$DATA_CFG" "$TRAIN_CFG" "$MODEL_CFG" <<'PY'
import os
import sys
import yaml

base_data, base_train, base_model, out_data, out_train, out_model = sys.argv[1:]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

data = load(base_data)
train = load(base_train)
model = load(base_model)

if os.environ.get("DATA_ROOT"):
    data["root"] = os.environ["DATA_ROOT"]
data["permute"] = os.environ.get("PERMUTE", "0") == "1"

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
  local cond="$1"; local seed="$2"; local gpu="$3"
  make_configs "$cond"
  local log="$RESULTS_DIR/logs/${TASK_LABEL}_${cond}_seed${seed}.log"
  local cmd="CUDA_VISIBLE_DEVICES=$gpu uv run python -m scripts.pilot.train_pilot \
    --task $TASK_LABEL --condition $cond --seed $seed \
    --project_config configs/default_project_configs.yaml \
    --data_config $DATA_CFG \
    --train_config $TRAIN_CFG \
    --model_config $MODEL_CFG \
    --optimizer_config $OPT_CFG \
    --results_dir $RESULTS_DIR"
  echo "[seqmnist] gpu=$gpu task=$TASK_LABEL cond=$cond seed=$seed -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  $cmd"
  else
    eval "$cmd" >"$log" 2>&1
  fi
}

i=0
declare -a PIDS=()
for cond in $CONDITIONS; do
  gpu="${GPU_ARR[$(( i % N_GPUS ))]}"
  i=$(( i + 1 ))
  (
    for seed in $SEEDS; do
      run_one "$cond" "$seed" "$gpu"
    done
  ) &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "[seqmnist] one runner returned non-zero" >&2
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[seqmnist] DRY_RUN=1, skipping aggregation"
else
  uv run python -m scripts.pilot.aggregate_results \
    --results_dir "$RESULTS_DIR" \
    --tasks "$TASK_LABEL" \
    --conditions "$CONDITIONS" \
    --seeds "$SEEDS"
fi
