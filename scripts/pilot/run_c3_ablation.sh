#!/usr/bin/env bash
# C3/MGA ablations for the follow-up study.
#
# Covers membrane-vs-spike γ input, γ/β roles, k_bits, and write_scale.
#
# Examples:
#   GPUS=0,1 RESULTS_DIR=pilot_results_c3_ablation bash scripts/pilot/run_c3_ablation.sh
#   T=200 SEEDS="42 123 2024" ABLATIONS="full gamma_spike k4" bash scripts/pilot/run_c3_ablation.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-shd}"        # shd or dvs128
T="${T:-100}"
SEEDS="${SEEDS:-42}"
GPUS="${GPUS:-0}"
RESULTS_DIR="${RESULTS_DIR:-pilot_results_c3_ablation}"
ABLATIONS="${ABLATIONS:-full gamma_spike gamma_rate gamma_only beta_only no_gates k2 k4 no_write_scale}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
DATA_ROOT="${DATA_ROOT:-}"
export DATA_ROOT MAX_EPOCHS BATCH_SIZE

mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/generated_configs"

case "$TASK" in
  shd)
    BASE_DATA_CFG="configs/data_configs/shd_pilot.yaml"
    BASE_TRAIN_CFG="configs/train_configs/pilot_shd.yaml"
    OPT_CFG="configs/optimizer_configs/pilot_shd.yaml"
    BASE_MODEL_CFG="configs/model_configs/pilot/spikformer_shd_c3_mga.yaml"
    ;;
  dvs128)
    BASE_DATA_CFG="configs/data_configs/dvs128gesture_pilot.yaml"
    BASE_TRAIN_CFG="configs/train_configs/pilot_dvs.yaml"
    OPT_CFG="configs/optimizer_configs/pilot_dvs.yaml"
    BASE_MODEL_CFG="configs/model_configs/pilot/spikformer_dvs_c3_mga.yaml"
    ;;
  *)
    echo "Unsupported TASK=$TASK (expected shd or dvs128)" >&2
    exit 1
    ;;
esac

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"

condition_label() {
  local ab="$1"
  [[ "$ab" == "full" ]] && echo "c3_mga" || echo "c3_mga_${ab}"
}

make_configs() {
  local ablation="$1"; local out_dir="$RESULTS_DIR/generated_configs/${TASK}_T${T}"
  mkdir -p "$out_dir"
  DATA_CFG="$out_dir/data.yaml"
  TRAIN_CFG="$out_dir/train.yaml"
  MODEL_CFG="$out_dir/model_c3_${ablation}.yaml"
  python - "$BASE_DATA_CFG" "$BASE_TRAIN_CFG" "$BASE_MODEL_CFG" "$DATA_CFG" "$TRAIN_CFG" "$MODEL_CFG" "$T" "$ablation" <<'PY'
import os
import sys
import yaml

base_data, base_train, base_model, out_data, out_train, out_model, T, ablation = sys.argv[1:]
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
model["attention_type"] = "c3_mga"

if ablation == "full":
    pass
elif ablation == "gamma_spike":
    model["mga_gamma_input"] = "spike"
elif ablation == "gamma_rate":
    model["mga_gamma_input"] = "rate"
elif ablation == "gamma_only":
    model["mga_use_beta"] = False
elif ablation == "beta_only":
    model["mga_use_gamma"] = False
elif ablation == "no_gates":
    model["mga_use_gamma"] = False
    model["mga_use_beta"] = False
elif ablation == "k2":
    model["mga_k_bits"] = 2
elif ablation == "k4":
    model["mga_k_bits"] = 4
elif ablation == "no_write_scale":
    model["mga_use_write_scale"] = False
else:
    raise SystemExit(f"unknown ablation: {ablation}")

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
  local ablation="$1"; local seed="$2"; local gpu="$3"
  make_configs "$ablation"
  local cond; cond="$(condition_label "$ablation")"
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
  echo "[C3-ablation] gpu=$gpu task=$task_label cond=$cond seed=$seed -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  $cmd"
  else
    eval "$cmd" >"$log" 2>&1
  fi
}

i=0
declare -a PIDS=()
for ablation in $ABLATIONS; do
  gpu="${GPU_ARR[$(( i % N_GPUS ))]}"
  i=$(( i + 1 ))
  (
    for seed in $SEEDS; do
      run_one "$ablation" "$seed" "$gpu"
    done
  ) &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "[C3-ablation] one runner returned non-zero" >&2
done

CONDITION_LABELS=""
for ablation in $ABLATIONS; do CONDITION_LABELS+="$(condition_label "$ablation") "; done
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[C3-ablation] DRY_RUN=1, skipping aggregation"
else
  uv run python -m scripts.pilot.aggregate_results \
    --results_dir "$RESULTS_DIR" \
    --tasks "${TASK}_T${T}" \
    --conditions "$CONDITION_LABELS" \
    --seeds "$SEEDS"
fi
