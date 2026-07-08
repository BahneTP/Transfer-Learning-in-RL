#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-0}"
RUN_ROOT="${2:-$(pwd)/logs/tl}"
RESUME_BATCH_DIR="${3:-${RESUME_BATCH_DIR:-}}"
SEEDS=(${SEEDS:-1 2 3 4 5})
GAME="jamesbond"
EXPERIMENT="atari100k/der/${GAME}_resnet_full"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${RESUME_BATCH_DIR}" ]]; then
  BATCH_DIR="${RESUME_BATCH_DIR}"
  BATCH_NAME="$(basename "${BATCH_DIR}")"
else
  BATCH_NAME="tl_der_jamesbond_resnet_full_lr_compare_$(date +%Y-%m-%d_%H-%M-%S)"
  BATCH_DIR="${RUN_ROOT}/${BATCH_NAME}"
fi
BATCH_LOG="${BATCH_DIR}/batch.log"
BATCH_RESULTS="${BATCH_DIR}/batch.results.tsv"
mkdir -p "${BATCH_DIR}"

if [[ ! -f "${BATCH_RESULTS}" ]]; then
  printf "group\tvariant\talgorithm\tgame\tseed\treturn_mean\treturn_std\treturn_min\treturn_max\tlog\n" \
    > "${BATCH_RESULTS}"
fi

is_completed() {
  local variant="$1"
  local seed="$2"

  awk -F '\t' \
    -v variant="${variant}" \
    -v seed="${seed}" \
    'NR > 1 && $1 == "resnet_full" && $2 == variant && $3 == "der" && $4 == "jamesbond" && $5 == seed && $6 != "NA" { found = 1 }
     END { exit found ? 0 : 1 }' \
    "${BATCH_RESULTS}"
}

run_one() {
  local variant="$1"
  local seed="$2"
  shift 2
  local overrides=("$@")

  local run_name="resnet_full_${variant}_der_${GAME}_seed${seed}"
  local out_dir="${BATCH_DIR}/${run_name}"
  local log_file="${out_dir}/train_eval.log"

  if is_completed "${variant}" "${seed}"; then
    echo "Skipping completed run: resnet_full | ${variant} | der | ${GAME} | seed ${seed}"
    return
  fi

  local cmd=(
    uv run python src/train.py
    "experiment=${EXPERIMENT}"
    "trainer.accelerator=gpu"
    "trainer.devices=[${DEVICE}]"
    "trainer.seed=${seed}"
    "logger=[]"
    "checkpoint.enabled=false"
    "hydra.run.dir=${out_dir}"
  )
  cmd+=("${overrides[@]}")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Would run: resnet_full | ${variant} | der | ${GAME} | seed ${seed}"
    echo "Command: ${cmd[*]}"
    return
  fi

  mkdir -p "${out_dir}"
  : > "${log_file}"

  {
    echo "=== resnet_full | ${variant} | der | ${GAME} | seed ${seed} ==="
    echo "Command: ${cmd[*]}"
    echo "Started at: $(date --iso-8601=seconds)"
    echo
    "${cmd[@]}"
    echo
    echo "Finished at: $(date --iso-8601=seconds)"
  } 2>&1 | tee -a "${log_file}"

  local return_mean="NA"
  local return_std="NA"
  local return_min="NA"
  local return_max="NA"
  return_mean="$(awk '/eval\/return_mean:/ {print $2}' "${log_file}" | tail -n 1)"
  return_std="$(awk '/eval\/return_std:/ {print $2}' "${log_file}" | tail -n 1)"
  return_min="$(awk '/eval\/return_min:/ {print $2}' "${log_file}" | tail -n 1)"
  return_max="$(awk '/eval\/return_max:/ {print $2}' "${log_file}" | tail -n 1)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "resnet_full" "${variant}" "der" "${GAME}" "${seed}" \
    "${return_mean:-NA}" "${return_std:-NA}" "${return_min:-NA}" "${return_max:-NA}" "${log_file}" \
    >> "${BATCH_RESULTS}"
}

main() {
  echo "Running DER ResNet full fine-tuning LR comparison on ${GAME}"
  echo "GPU: ${DEVICE}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Batch directory: ${BATCH_DIR}"
  if [[ -n "${RESUME_BATCH_DIR}" ]]; then
    echo "Resume mode: skipping completed rows in ${BATCH_RESULTS}"
  fi
  echo

  for seed in "${SEEDS[@]}"; do
    run_one "enc1e-7_algo1e-4" "${seed}" \
      "algorithm.learning_rate=1e-4" \
      "algorithm.encoder_lr_scale=0.001"

    run_one "uniform1e-5" "${seed}" \
      "algorithm.learning_rate=1e-5" \
      "algorithm.encoder_lr_scale=1.0"
  done

  echo
  echo "Finished batch at: $(date --iso-8601=seconds)"
  echo "Results: ${BATCH_RESULTS}"
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  main
else
  main 2>&1 | tee -a "${BATCH_LOG}"
fi
