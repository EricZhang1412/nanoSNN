#!/usr/bin/env bash
# Gate-1 pilot launcher.
#
# Schedules 24 training runs (4 conditions × 3 seeds × 2 tasks).  Conditions
# are dispatched in parallel across GPUs; seeds within a condition run
# sequentially.  Set GPUS to a comma-separated list of CUDA devices, e.g.
#   GPUS=0,1,2,3 bash scripts/pilot/run_all.sh
#
# Optional env vars:
#   RESULTS_DIR  (default: pilot_results)
#   TASKS        (default: "dvs128 shd"; can be "dvs128" or "shd")
#   CONDITIONS   (default: "c0_sdla c1_lowrank c2_oneminusk c3_mga")
#   SEEDS        (default: "42 123 2024")
#   DRY_RUN      (default: 0; set to 1 to just print the command lines)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0}"
RESULTS_DIR="${RESULTS_DIR:-pilot_results}"
TASKS="${TASKS:-dvs128 shd}"
CONDITIONS="${CONDITIONS:-c0_sdla c1_lowrank c2_oneminusk c3_mga}"
SEEDS="${SEEDS:-42 123 2024}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/logs"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"

# Map a (task, condition) pair to one of: project, data, train, model, optimizer configs.
make_paths() {
  local task="$1"; local cond="$2"
  case "$task" in
    dvs128)
      DATA_CFG="configs/data_configs/dvs128gesture_pilot.yaml"
      TRAIN_CFG="configs/train_configs/pilot_dvs.yaml"
      OPT_CFG="configs/optimizer_configs/pilot_dvs.yaml"
      MODEL_CFG="configs/model_configs/pilot/spikformer_dvs_${cond}.yaml"
      ;;
    shd)
      DATA_CFG="configs/data_configs/shd_pilot.yaml"
      TRAIN_CFG="configs/train_configs/pilot_shd.yaml"
      OPT_CFG="configs/optimizer_configs/pilot_shd.yaml"
      MODEL_CFG="configs/model_configs/pilot/spikformer_shd_${cond}.yaml"
      ;;
    *)
      echo "unknown task: $task" >&2; exit 1 ;;
  esac
  PROJ_CFG="configs/default_project_configs.yaml"
}

run_one() {
  local task="$1"; local cond="$2"; local seed="$3"; local gpu="$4"
  make_paths "$task" "$cond"
  local log="$RESULTS_DIR/logs/${task}_${cond}_seed${seed}.log"
  local cmd="CUDA_VISIBLE_DEVICES=$gpu uv run python -m scripts.pilot.train_pilot \
    --task $task --condition $cond --seed $seed \
    --project_config $PROJ_CFG \
    --data_config   $DATA_CFG  \
    --train_config  $TRAIN_CFG \
    --model_config  $MODEL_CFG \
    --optimizer_config $OPT_CFG \
    --results_dir $RESULTS_DIR"
  echo "[pilot]  gpu=$gpu  $task  $cond  seed=$seed  →  $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  $cmd"
  else
    eval "$cmd" >"$log" 2>&1
  fi
}

# Round-robin conditions across GPUs.  Within a (gpu, task, condition), iterate seeds sequentially.
i=0
declare -a PIDS=()
for task in $TASKS; do
  for cond in $CONDITIONS; do
    gpu="${GPU_ARR[$(( i % N_GPUS ))]}"
    i=$(( i + 1 ))
    (
      for seed in $SEEDS; do
        run_one "$task" "$cond" "$seed" "$gpu"
      done
    ) &
    PIDS+=($!)
  done
done

# Wait for all background runners.
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "[pilot] one runner returned non-zero" >&2
done

echo "[pilot] all runs finished. Aggregating..."
uv run python -m scripts.pilot.aggregate_results --results_dir "$RESULTS_DIR"
echo "[pilot] done."
