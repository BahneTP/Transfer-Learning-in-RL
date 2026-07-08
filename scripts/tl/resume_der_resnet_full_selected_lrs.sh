#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-0}"
BATCH_DIR="${2:-logs/tl/tl_bbf_der_resnet_full_2026-07-01_09-57-02}"
SEEDS=(${SEEDS:-1 2 3 4 5})
GAMES=(${GAMES:-jamesbond assault bankheist roadrunner})

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

BATCH_LOG="${BATCH_DIR}/batch.log"
BATCH_RESULTS="${BATCH_DIR}/batch.results.tsv"
mkdir -p "${BATCH_DIR}"

if [[ ! -f "${BATCH_RESULTS}" ]]; then
  printf "group\tvariant\talgorithm\tgame\tseed\treturn_mean\treturn_std\treturn_min\treturn_max\tlog\n" \
    > "${BATCH_RESULTS}"
fi

is_completed() {
  local variant="$1"
  local game="$2"
  local seed="$3"

  awk -F '\t' \
    -v variant="${variant}" \
    -v game="${game}" \
    -v seed="${seed}" \
    'NR > 1 && $1 == "resnet_full" && $2 == variant && $3 == "der" && $4 == game && $5 == seed && $6 != "NA" { found = 1 }
     END { exit found ? 0 : 1 }' \
    "${BATCH_RESULTS}"
}

run_one() {
  local variant="$1"
  local scale="$2"
  local game="$3"
  local seed="$4"

  local experiment="atari100k/der/${game}_resnet_full"
  local run_name="resnet_full_${variant}_der_${game}_seed${seed}"
  local out_dir="${BATCH_DIR}/${run_name}"
  local log_file="${out_dir}/train_eval.log"

  if is_completed "${variant}" "${game}" "${seed}"; then
    echo "Skipping completed run: resnet_full | ${variant} | der | ${game} | seed ${seed}"
    return
  fi

  local cmd=(
    uv run python src/train.py
    "experiment=${experiment}"
    "trainer.accelerator=gpu"
    "trainer.devices=[${DEVICE}]"
    "trainer.seed=${seed}"
    "logger=[]"
    "checkpoint.enabled=false"
    "hydra.run.dir=${out_dir}"
    "algorithm.encoder_lr_scale=${scale}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Would run: resnet_full | ${variant} | der | ${game} | seed ${seed}"
    echo "Command: ${cmd[*]}"
    return
  fi

  mkdir -p "${out_dir}"
  : > "${log_file}"

  {
    echo "=== resnet_full | ${variant} | der | ${game} | seed ${seed} ==="
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
    "resnet_full" "${variant}" "der" "${game}" "${seed}" \
    "${return_mean:-NA}" "${return_std:-NA}" "${return_min:-NA}" "${return_max:-NA}" "${log_file}" \
    >> "${BATCH_RESULTS}"
}

main() {
  echo "Resuming selected DER ResNet full fine-tuning runs"
  echo "GPU: ${DEVICE}"
  echo "Batch directory: ${BATCH_DIR}"
  echo "Games: ${GAMES[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Variants: enc_lr_1e-6, enc_lr_1e-5, enc_lr_1e-4"
  echo

  for game in "${GAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "enc_lr_1e-6" "0.01" "${game}" "${seed}"
      run_one "enc_lr_1e-5" "0.1" "${game}" "${seed}"
      run_one "enc_lr_1e-4" "1.0" "${game}" "${seed}"
    done
  done

  echo
  echo "Finished selected resume at: $(date --iso-8601=seconds)"
  echo "Results: ${BATCH_RESULTS}"
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  main
else
  main 2>&1 | tee -a "${BATCH_LOG}"
fi
