#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

GPU="${GPU:-}"
GAMES="${GAMES:-qbert battlezone}"
SEEDS="${SEEDS:-1 2 3 4 5}"
STEPS="${STEPS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
LOG_EVERY="${LOG_EVERY:-1000}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/logs/atari100k_rainbow_batch_$(date +%Y-%m-%d_%H-%M-%S)}"

if [[ -z "${GPU}" ]]; then
  GPU="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}' \
      | sort -k2,2nr \
      | head -n 1 \
      | awk '{print $1}'
  )"
fi

mkdir -p "${RUN_ROOT}"
RESULTS_FILE="${RUN_ROOT}/results.tsv"
printf "algorithm\tgame\tseed\tgpu\tsteps\teval_episodes\teval_return_mean\teval_return_std\teval_return_min\teval_return_max\ttrain_log\teval_log\n" > "${RESULTS_FILE}"

metric_from_log() {
  local metric="$1"
  local log_file="$2"
  awk -F': ' -v key="${metric}" '$1 ~ key {print $2}' "${log_file}" | tail -n 1
}

run_one() {
  local game="$1"
  local seed="$2"
  local experiment="rainbow/${game}_atari100k"
  local run_dir="${RUN_ROOT}/rainbow_${game}_seed${seed}"
  local train_dir="${run_dir}/train"
  local eval_dir="${run_dir}/eval"
  local train_log="${run_dir}/train.log"
  local eval_log="${run_dir}/eval.log"
  local checkpoint="${train_dir}/checkpoints/last.pt"

  mkdir -p "${run_dir}"
  echo "Running rainbow ${game}: seed=${seed}, steps=${STEPS}, eval_games=${EVAL_EPISODES}, gpu=${GPU}"

  if [[ -f "${checkpoint}" ]]; then
    echo "  Reusing checkpoint: ${checkpoint}"
  else
    python -u src/train.py \
      "experiment=${experiment}" \
      "trainer.seed=${seed}" \
      "trainer.total_frames=${STEPS}" \
      "trainer.log_every_n_steps=${LOG_EVERY}" \
      "trainer.accelerator=gpu" \
      "trainer.devices=[${GPU}]" \
      "logger=[]" \
      "checkpoint.save_every_n_steps=999999999" \
      "checkpoint.save_last=true" \
      "hydra.run.dir=${train_dir}" \
      > "${train_log}" 2>&1
  fi

  python -u src/eval.py \
    "algorithm=rainbow_atari100k" \
    "environment=${game}_atari100k_train" \
    "environment@eval_environment=${game}_atari100k_eval" \
    "trainer.seed=${seed}" \
    "trainer.num_eval_episodes=${EVAL_EPISODES}" \
    "trainer.accelerator=gpu" \
    "trainer.devices=[${GPU}]" \
    "logger=[]" \
    "checkpoint.resume_from=${checkpoint}" \
    "hydra.run.dir=${eval_dir}" \
    > "${eval_log}" 2>&1

  local mean std min max
  mean="$(metric_from_log "eval/return_mean" "${eval_log}")"
  std="$(metric_from_log "eval/return_std" "${eval_log}")"
  min="$(metric_from_log "eval/return_min" "${eval_log}")"
  max="$(metric_from_log "eval/return_max" "${eval_log}")"

  printf "rainbow\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${game}" "${seed}" "${GPU}" "${STEPS}" "${EVAL_EPISODES}" \
    "${mean:-NA}" "${std:-NA}" "${min:-NA}" "${max:-NA}" \
    "${train_log}" "${eval_log}" >> "${RESULTS_FILE}"
}

print_stats() {
  python - "${RESULTS_FILE}" <<'PY'
from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8"), delimiter="\t"))
by_game: dict[str, list[float]] = defaultdict(list)
for row in rows:
    try:
        by_game[row["game"]].append(float(row["eval_return_mean"]))
    except ValueError:
        pass

print()
print("Per-run results:")
print("game\tseed\teval_mean\teval_std\teval_min\teval_max")
for row in rows:
    print(
        f"{row['game']}\t{row['seed']}\t{row['eval_return_mean']}\t"
        f"{row['eval_return_std']}\t{row['eval_return_min']}\t{row['eval_return_max']}"
    )

print()
print("Statistics per game:")
print("game\tn\tmean\tstd\tmin\tmax")
for game in sorted(by_game):
    values = by_game[game]
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"{game}\t{len(values)}\t{statistics.mean(values):.4f}\t"
        f"{std:.4f}\t{min(values):.4f}\t{max(values):.4f}"
    )

missing = len(rows) - sum(len(v) for v in by_game.values())
if missing:
    print(f"\nWarning: {missing} run(s) had no parseable eval/return_mean.", file=sys.stderr)
PY
}

on_error() {
  local exit_code=$?
  echo "Batch failed. Partial results live in: ${RUN_ROOT}" >&2
  local latest_log
  latest_log="$(find "${RUN_ROOT}" -name "*.log" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  if [[ -n "${latest_log}" ]]; then
    echo "Last log tail: ${latest_log}" >&2
    tail -n 80 "${latest_log}" >&2
  fi
  exit "${exit_code}"
}
trap on_error ERR

echo "Batch root: ${RUN_ROOT}"
echo "GPU: ${GPU}"
echo "Games: ${GAMES}"
echo "Seeds: ${SEEDS}"
echo

for game in ${GAMES}; do
  for seed in ${SEEDS}; do
    run_one "${game}" "${seed}"
  done
done

echo
echo "Finished. Raw results: ${RESULTS_FILE}"
print_stats
