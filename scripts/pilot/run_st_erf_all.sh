#!/usr/bin/env bash
# Run ST-ERF diagnostic on seed-42 checkpoints of all four conditions, both tasks.
# Per pilot §5, only seed 42 is used; 256 samples each.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="${RESULTS_DIR:-pilot_results}"
N_SAMPLES="${N_SAMPLES:-256}"
GPU="${GPU:-0}"

for task in dvs128 shd; do
  case "$task" in
    dvs128)
      DATA_CFG="configs/data_configs/dvs128gesture_pilot.yaml"
      ;;
    shd)
      DATA_CFG="configs/data_configs/shd_pilot.yaml"
      ;;
  esac
  for cond in c0_sdla c1_lowrank c2_oneminusk c3_mga; do
    MODEL_CFG="configs/model_configs/pilot/spikformer_${task/dvs128/dvs}_${cond}.yaml"
    [[ "$task" == "shd" ]] && MODEL_CFG="configs/model_configs/pilot/spikformer_shd_${cond}.yaml"
    [[ "$task" == "dvs128" ]] && MODEL_CFG="configs/model_configs/pilot/spikformer_dvs_${cond}.yaml"
    CKPT="$RESULTS_DIR/checkpoints/$task/$cond/seed42/last.ckpt"
    if [[ ! -f "$CKPT" ]]; then
      echo "[skip] no checkpoint at $CKPT"
      continue
    fi
    echo "[st-erf] $task / $cond"
    CUDA_VISIBLE_DEVICES="$GPU" uv run python -m scripts.pilot.st_erf_diag \
      --ckpt "$CKPT" --task "$task" --condition "$cond" \
      --data_config "$DATA_CFG" --model_config "$MODEL_CFG" \
      --out_dir "$RESULTS_DIR/figs" --n_samples "$N_SAMPLES"
  done
done
