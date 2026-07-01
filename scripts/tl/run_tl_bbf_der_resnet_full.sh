#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-0}"
RUN_ROOT="${2:-$(pwd)/logs/tl}"
SEEDS=(${SEEDS:-1 2 3 4 5})
GAMES=(jamesbond assault bankheist roadrunner)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

BATCH_NAME="tl_bbf_der_resnet_full_$(date +%Y-%m-%d_%H-%M-%S)"
BATCH_DIR="${RUN_ROOT}/${BATCH_NAME}"
BATCH_LOG="${BATCH_DIR}/batch.log"
BATCH_RESULTS="${BATCH_DIR}/batch.results.tsv"
mkdir -p "${BATCH_DIR}"

printf "group\tvariant\talgorithm\tgame\tseed\treturn_mean\treturn_std\treturn_min\treturn_max\tlog\n" \
  > "${BATCH_RESULTS}"

run_one() {
  local group="$1"
  local variant="$2"
  local algo="$3"
  local game="$4"
  local seed="$5"
  local experiment="$6"
  shift 6
  local overrides=("$@")

  local run_name="${group}_${variant}_${algo}_${game}_seed${seed}"
  local out_dir="${BATCH_DIR}/${run_name}"
  local log_file="${out_dir}/train_eval.log"
  mkdir -p "${out_dir}"
  : > "${log_file}"

  local cmd=(
    uv run python src/train.py
    "experiment=${experiment}"
    "trainer.accelerator=gpu"
    "trainer.devices=[${DEVICE}]"
    "trainer.seed=${seed}"
    "logger=[]"
    "checkpoint.save_dir=${out_dir}/checkpoints"
    "hydra.run.dir=${out_dir}"
  )
  cmd+=("${overrides[@]}")

  {
    echo "=== ${group} | ${variant} | ${algo} | ${game} | seed ${seed} ==="
    echo "Command: ${cmd[*]}"
    echo "Started at: $(date --iso-8601=seconds)"
    echo
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN=1, not executing."
    else
      "${cmd[@]}"
    fi
    echo
    echo "Finished at: $(date --iso-8601=seconds)"
  } 2>&1 | tee -a "${log_file}"

  local return_mean="NA"
  local return_std="NA"
  local return_min="NA"
  local return_max="NA"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    return_mean="$(awk '/eval\/return_mean:/ {print $2}' "${log_file}" | tail -n 1)"
    return_std="$(awk '/eval\/return_std:/ {print $2}' "${log_file}" | tail -n 1)"
    return_min="$(awk '/eval\/return_min:/ {print $2}' "${log_file}" | tail -n 1)"
    return_max="$(awk '/eval\/return_max:/ {print $2}' "${log_file}" | tail -n 1)"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${group}" "${variant}" "${algo}" "${game}" "${seed}" \
    "${return_mean:-NA}" "${return_std:-NA}" "${return_min:-NA}" "${return_max:-NA}" "${log_file}" \
    >> "${BATCH_RESULTS}"
}

{
  echo "Running TL batch 1: BBF normal, DER normal, DER ResNet full fine-tune LR sweep"
  echo "GPU: ${DEVICE}"
  echo "Games: ${GAMES[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Batch directory: ${BATCH_DIR}"
  echo

  for game in "${GAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "baseline" "normal" "bbf" "${game}" "${seed}" "atari100k/bbf/${game}"
      run_one "baseline" "normal" "der" "${game}" "${seed}" "atari100k/der/${game}"

      run_one "resnet_full" "enc_lr_1e-6" "der" "${game}" "${seed}" \
        "atari100k/der/${game}_resnet_full" "algorithm.encoder_lr_scale=0.01"
      run_one "resnet_full" "enc_lr_3e-6" "der" "${game}" "${seed}" \
        "atari100k/der/${game}_resnet_full" "algorithm.encoder_lr_scale=0.03"
      run_one "resnet_full" "enc_lr_1e-5" "der" "${game}" "${seed}" \
        "atari100k/der/${game}_resnet_full" "algorithm.encoder_lr_scale=0.1"
      run_one "resnet_full" "enc_lr_3e-5" "der" "${game}" "${seed}" \
        "atari100k/der/${game}_resnet_full" "algorithm.encoder_lr_scale=0.3"
      run_one "resnet_full" "enc_lr_1e-4" "der" "${game}" "${seed}" \
        "atari100k/der/${game}_resnet_full" "algorithm.encoder_lr_scale=1.0"
      run_one "resnet_full" "enc_lr_3e-4" "der" "${game}" "${seed}" \
        "atari100k/der/${game}_resnet_full" "algorithm.encoder_lr_scale=3.0"
    done
  done

  echo
  echo "Finished batch at: $(date --iso-8601=seconds)"
  echo "Results: ${BATCH_RESULTS}"
} 2>&1 | tee -a "${BATCH_LOG}"
