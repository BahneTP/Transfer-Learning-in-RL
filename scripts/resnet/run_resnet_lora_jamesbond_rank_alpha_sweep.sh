#!/usr/bin/env bash
set -uo pipefail

GPU="${1:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
RANK_ALPHA_PAIRS="${RANK_ALPHA_PAIRS:-1:2 2:4 4:8 8:16 16:32}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/resnet/resnet_lora_jamesbond_rank_alpha_sweep_${STAMP}}"
RESULTS_CSV="${RESULTS_CSV:-${RUN_ROOT}/results.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${RUN_ROOT}/summary.csv}"

mkdir -p "${RUN_ROOT}/logs"

echo "rank,alpha,seed,status,eval_return_mean,eval_return_std,eval_return_min,eval_return_max,eval_num_episodes,time_eval,log_file" > "${RESULTS_CSV}"

append_result() {
  local rank="$1"
  local alpha="$2"
  local seed="$3"
  local status="$4"
  local log_file="$5"
  uv run python - "$RESULTS_CSV" "$rank" "$alpha" "$seed" "$status" "$log_file" "$EVAL_EPISODES" <<'PY'
import csv
import re
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
rank = sys.argv[2]
alpha = sys.argv[3]
seed = sys.argv[4]
status = sys.argv[5]
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
        rank,
        alpha,
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
  uv run python - "$RESULTS_CSV" "$SUMMARY_CSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

results_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
groups = defaultdict(list)

if results_path.exists():
    with results_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "0":
                continue
            value = row.get("eval_return_mean", "")
            if value:
                groups[(row["rank"], row["alpha"])].append(float(value))

with summary_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "alpha", "metric", "n", "mean", "std", "min", "max"])
    for (rank, alpha), values in sorted(groups.items(), key=lambda item: (int(item[0][0]), float(item[0][1]))):
        writer.writerow([
            rank,
            alpha,
            "eval_return_mean",
            len(values),
            statistics.mean(values),
            statistics.pstdev(values),
            min(values),
            max(values),
        ])
    if not groups:
        writer.writerow(["", "", "eval_return_mean", 0, "", "", "", ""])
PY
}

for pair in ${RANK_ALPHA_PAIRS}; do
  rank="${pair%%:*}"
  alpha="${pair#*:}"
  for seed in ${SEEDS}; do
    LOG_FILE="${RUN_ROOT}/logs/r${rank}_a${alpha}_seed_${seed}.log"
    HYDRA_DIR="${RUN_ROOT}/hydra/r${rank}_a${alpha}/seed_${seed}"
    RUN_NAME="resnet_lora_jamesbond_r${rank}_a${alpha}_seed_${seed}_${STAMP}"

    echo "Starting rank ${rank}, alpha ${alpha}, seed ${seed} on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" uv run python src/train.py \
      experiment=resnet/der/lora/jamesbond \
      trainer.seed="${seed}" \
      trainer.devices="[0]" \
      trainer.num_eval_episodes="${EVAL_EPISODES}" \
      hydra.run.dir="${HYDRA_DIR}" \
      logger.0.name="${RUN_NAME}" \
      logger.0.save_dir="${RUN_ROOT}/wandb" \
      checkpoint.enabled=false \
      checkpoint.save_last=false \
      checkpoint.save_every_n_steps=999999999 \
      algorithm.lora_rank="${rank}" \
      algorithm.lora_alpha="${alpha}" \
      2>&1 | tee "${LOG_FILE}"
    status="${PIPESTATUS[0]}"
    append_result "${rank}" "${alpha}" "${seed}" "${status}" "${LOG_FILE}"

    if [[ "${status}" -ne 0 ]]; then
      echo "Rank ${rank}, alpha ${alpha}, seed ${seed} failed with status ${status}. Continuing." >&2
    fi
  done
done

write_summary
echo "Results written to ${RESULTS_CSV}"
echo "Summary written to ${SUMMARY_CSV}"
