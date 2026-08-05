#!/usr/bin/env bash
set -uo pipefail

GPU="${1:-0}"
SEED="${SEED:-1}"
WEIGHTS="${DINO_WEIGHTS:-models/dinov2_vits14_pretrain.pth}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LAMBDA_SIGREG="${LAMBDA_SIGREG:-0.1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/jepa_comparison/dino_jamesbond_full_block3_sigreg_${LAMBDA_SIGREG//./p}_checkpoint_${STAMP}}"
RESULTS_CSV="${RESULTS_CSV:-${RUN_ROOT}/results.csv}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints"

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing DINOv2 weights: ${WEIGHTS}" >&2
  exit 1
fi

echo "variant,seed,status,eval_return_mean,eval_return_std,eval_return_min,eval_return_max,eval_num_episodes,time_eval,checkpoint,log_file" > "${RESULTS_CSV}"

append_result() {
  local variant="$1"
  local seed="$2"
  local status="$3"
  local checkpoint="$4"
  local log_file="$5"
  python - "$RESULTS_CSV" "$variant" "$seed" "$status" "$checkpoint" "$log_file" "$EVAL_EPISODES" <<'PY'
import csv
import re
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
variant = sys.argv[2]
seed = sys.argv[3]
status = sys.argv[4]
checkpoint = sys.argv[5]
log_file = Path(sys.argv[6])
eval_episodes = sys.argv[7]

metrics = {
    "eval/return_mean": "",
    "eval/return_std": "",
    "eval/return_min": "",
    "eval/return_max": "",
    "eval/num_episodes": eval_episodes,
    "time/eval": "",
}

pattern = re.compile(r"^\s*(eval/(?:return_mean|return_std|return_min|return_max|num_episodes)|time/eval):\s*([-+0-9.eE]+)")
if log_file.exists():
    for line in log_file.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            metrics[match.group(1)] = match.group(2)

with csv_path.open("a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        variant,
        seed,
        status,
        metrics["eval/return_mean"],
        metrics["eval/return_std"],
        metrics["eval/return_min"],
        metrics["eval/return_max"],
        metrics["eval/num_episodes"],
        metrics["time/eval"],
        checkpoint,
        str(log_file),
    ])
PY
}

VARIANT="full_block3_sigreg_${LAMBDA_SIGREG//./p}"
LOG_FILE="${RUN_ROOT}/logs/${VARIANT}_seed_${SEED}.log"
HYDRA_DIR="${RUN_ROOT}/hydra/${VARIANT}/seed_${SEED}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints/${VARIANT}/seed_${SEED}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/last.pt"
RUN_NAME="dino_jamesbond_${VARIANT}_seed_${SEED}_${STAMP}"

mkdir -p "${CHECKPOINT_DIR}"

echo "Starting ${VARIANT} seed ${SEED} on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" uv run python src/train.py \
  experiment=dinov2/der/jepa_full_block3/jamesbond \
  algorithm.dinov2_weights="${WEIGHTS}" \
  algorithm.jepa_loss_weight=0.0 \
  algorithm.jepa_prediction_mode=residual \
  algorithm.temporal_straightening_weight=0.0 \
  algorithm.lambda_sigreg="${LAMBDA_SIGREG}" \
  trainer.seed="${SEED}" \
  trainer.devices="[0]" \
  trainer.num_eval_episodes="${EVAL_EPISODES}" \
  hydra.run.dir="${HYDRA_DIR}" \
  logger.0.name="${RUN_NAME}" \
  logger.0.save_dir="${RUN_ROOT}/wandb" \
  checkpoint.enabled=true \
  checkpoint.save_dir="${CHECKPOINT_DIR}" \
  checkpoint.save_last=true \
  checkpoint.save_every_n_steps=999999999 \
  2>&1 | tee "${LOG_FILE}"
status="${PIPESTATUS[0]}"
append_result "${VARIANT}" "${SEED}" "${status}" "${CHECKPOINT_PATH}" "${LOG_FILE}"

if [[ "${status}" -ne 0 ]]; then
  echo "${VARIANT} failed with status ${status}." >&2
fi

echo "Results written to ${RESULTS_CSV}"
echo "Checkpoint expected at ${CHECKPOINT_PATH}"
exit "${status}"
