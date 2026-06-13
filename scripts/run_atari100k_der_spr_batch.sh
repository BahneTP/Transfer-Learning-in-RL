#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

TOTAL_FRAMES="${TOTAL_FRAMES:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/logs/atari100k_batch_$(date +%Y-%m-%d_%H-%M-%S)}"

if [[ -n "${GPU:-}" ]]; then
  GPU_ID="${GPU}"
else
  GPU_ID="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}' \
      | sort -k2,2nr \
      | head -n 1 \
      | awk '{print $1}'
  )"
fi

mkdir -p "${RUN_ROOT}"
RESULTS_FILE="${RUN_ROOT}/results.tsv"
printf "algorithm\tgame\tseed\tgpu\teval_return_mean\teval_return_std\teval_return_min\teval_return_max\ttrain_log\teval_log\n" > "${RESULTS_FILE}"

run_one() {
  local algorithm="$1"
  local game="$2"
  local seed="$3"
  local experiment="${algorithm}/${game}_atari100k"
  local run_dir="${RUN_ROOT}/${algorithm}_${game}_seed${seed}"
  local train_log="${run_dir}/train.log"
  local eval_log="${run_dir}/eval.log"
  local checkpoint="${run_dir}/train/checkpoints/last.pt"

  mkdir -p "${run_dir}"

  python -u src/train.py \
    "experiment=${experiment}" \
    "trainer.accelerator=gpu" \
    "trainer.devices=[${GPU_ID}]" \
    "trainer.seed=${seed}" \
    "trainer.total_frames=${TOTAL_FRAMES}" \
    "logger=[]" \
    "checkpoint.save_every_n_steps=999999999" \
    "checkpoint.save_last=true" \
    "hydra.run.dir=${run_dir}/train" \
    > "${train_log}" 2>&1

  python -u src/eval.py \
    "experiment=${experiment}" \
    "trainer.accelerator=gpu" \
    "trainer.devices=[${GPU_ID}]" \
    "trainer.seed=${seed}" \
    "trainer.num_eval_episodes=${EVAL_EPISODES}" \
    "logger=[]" \
    "checkpoint.resume_from=${checkpoint}" \
    "hydra.run.dir=${run_dir}/eval" \
    > "${eval_log}" 2>&1

  local mean std min max
  mean="$(awk -F': ' '/eval\/return_mean/ {print $2}' "${eval_log}" | tail -n 1)"
  std="$(awk -F': ' '/eval\/return_std/ {print $2}' "${eval_log}" | tail -n 1)"
  min="$(awk -F': ' '/eval\/return_min/ {print $2}' "${eval_log}" | tail -n 1)"
  max="$(awk -F': ' '/eval\/return_max/ {print $2}' "${eval_log}" | tail -n 1)"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${algorithm}" "${game}" "${seed}" "${GPU_ID}" \
    "${mean}" "${std}" "${min}" "${max}" \
    "${train_log}" "${eval_log}" >> "${RESULTS_FILE}"

  if [[ "${KEEP_CHECKPOINTS}" != "1" ]]; then
    rm -f "${checkpoint}"
  fi
}

on_error() {
  local exit_code=$?
  echo "Run failed. Partial logs and results are in: ${RUN_ROOT}" >&2
  echo "Last 80 lines from the newest log:" >&2
  find "${RUN_ROOT}" -name "*.log" -printf "%T@ %p\n" \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2- \
    | xargs -r tail -n 80 >&2
  exit "${exit_code}"
}
trap on_error ERR

for game in qbert battlezone; do
  for algorithm in der spr; do
    for seed in 1 2 3; do
      run_one "${algorithm}" "${game}" "${seed}"
    done
  done
done

echo "Finished Atari 100K batch on GPU ${GPU_ID}."
echo "Logs: ${RUN_ROOT}"
echo
column -t -s $'\t' "${RESULTS_FILE}"
