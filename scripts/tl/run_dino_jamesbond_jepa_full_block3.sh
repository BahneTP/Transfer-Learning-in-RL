#!/usr/bin/env bash
set -uo pipefail

GPU="${1:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
WEIGHTS="${DINO_WEIGHTS:-models/dinov2_vits14_pretrain.pth}"
JEPA_WEIGHT="${JEPA_WEIGHT:-1.0}"
JEPA_PREDICTION_MODE="${JEPA_PREDICTION_MODE:-direct}"
TEMPORAL_STRAIGHTENING_WEIGHT="${TEMPORAL_STRAIGHTENING_WEIGHT:-0.0}"
LAMBDA_SIGREG="${LAMBDA_SIGREG:-0.0}"
RUN_LABEL="${RUN_LABEL:-weight_${JEPA_WEIGHT//./p}}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/tl/dino_jamesbond_jepa_full_block3_${RUN_LABEL}_${STAMP}}"
RESULTS_CSV="${RESULTS_CSV:-${RUN_ROOT}/results.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${RUN_ROOT}/summary.csv}"

mkdir -p "${RUN_ROOT}/logs"

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing DINOv2 weights: ${WEIGHTS}" >&2
  exit 1
fi

echo "seed,status,eval_return_mean,eval_return_std,eval_return_min,eval_return_max,eval_num_episodes,time_eval,log_file" > "${RESULTS_CSV}"

append_result() {
  local seed="$1"
  local status="$2"
  local log_file="$3"
  python - "$RESULTS_CSV" "$seed" "$status" "$log_file" "$EVAL_EPISODES" <<'PY'
import csv
import re
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
seed = sys.argv[2]
status = sys.argv[3]
log_file = Path(sys.argv[4])
eval_episodes = sys.argv[5]

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
        seed,
        status,
        metrics["eval/return_mean"],
        metrics["eval/return_std"],
        metrics["eval/return_min"],
        metrics["eval/return_max"],
        metrics["eval/num_episodes"],
        metrics["time/eval"],
        str(log_file),
    ])
PY
}

write_summary() {
  python - "$RESULTS_CSV" "$SUMMARY_CSV" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
values = []

if results_path.exists():
    with results_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "0":
                continue
            value = row.get("eval_return_mean", "")
            if value:
                values.append(float(value))

with summary_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "n", "mean", "std", "min", "max"])
    if values:
        writer.writerow([
            "eval_return_mean",
            len(values),
            statistics.mean(values),
            statistics.pstdev(values),
            min(values),
            max(values),
        ])
    else:
        writer.writerow(["eval_return_mean", 0, "", "", "", ""])
PY
}

for seed in ${SEEDS}; do
  LOG_FILE="${RUN_ROOT}/logs/seed_${seed}.log"
  HYDRA_DIR="${RUN_ROOT}/hydra/seed_${seed}"
  RUN_NAME="dino_jamesbond_jepa_full_block3_${RUN_LABEL}_seed_${seed}_${STAMP}"

  echo "Starting seed ${seed} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" uv run python src/train.py \
    experiment=dinov2/der/jepa_full_block3/jamesbond \
    algorithm.dinov2_weights="${WEIGHTS}" \
    algorithm.jepa_loss_weight="${JEPA_WEIGHT}" \
    algorithm.jepa_prediction_mode="${JEPA_PREDICTION_MODE}" \
    algorithm.temporal_straightening_weight="${TEMPORAL_STRAIGHTENING_WEIGHT}" \
    algorithm.lambda_sigreg="${LAMBDA_SIGREG}" \
    trainer.seed="${seed}" \
    trainer.devices="[0]" \
    trainer.num_eval_episodes="${EVAL_EPISODES}" \
    hydra.run.dir="${HYDRA_DIR}" \
    logger.0.name="${RUN_NAME}" \
    logger.0.save_dir="${RUN_ROOT}/wandb" \
    checkpoint.enabled=false \
    checkpoint.save_last=false \
    checkpoint.save_every_n_steps=999999999 \
    2>&1 | tee "${LOG_FILE}"
  status="${PIPESTATUS[0]}"
  append_result "${seed}" "${status}" "${LOG_FILE}"

  if [[ "${status}" -ne 0 ]]; then
    echo "Seed ${seed} failed with status ${status}. Continuing with remaining seeds." >&2
  fi
done

write_summary
echo "Results written to ${RESULTS_CSV}"
echo "Summary written to ${SUMMARY_CSV}"
