#!/usr/bin/env bash
set -euo pipefail

ALGO="${1:-der}"
DEVICE="${2:-0}"
START_SEED="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "${ALGO}" in
  der|spr|sr_spr|bbf|sac_bbf|dummy) ;;
  *)
    echo "Unsupported algorithm: ${ALGO}" >&2
    echo "Usage: $0 [der|spr|sr_spr|bbf|sac_bbf|dummy] [gpu_id] [start_seed]" >&2
    exit 2
    ;;
esac

if ! [[ "${DEVICE}" =~ ^[0-9]+$ && "${START_SEED}" =~ ^[0-9]+$ ]]; then
  echo "GPU ID and start seed must be non-negative integers." >&2
  echo "Usage: $0 [der|spr|sr_spr|bbf|sac_bbf|dummy] [gpu_id] [start_seed]" >&2
  exit 2
fi

END_SEED=$((START_SEED + 4))
BATCH_NAME="atari100k_batch_${ALGO}_wandb_$(date +%Y-%m-%d_%H-%M-%S)"
BATCH_DIR="${PROJECT_ROOT}/logs/${BATCH_NAME}"
BATCH_LOG="${BATCH_DIR}/batch.log"
BATCH_RESULTS="${BATCH_DIR}/batch.results.log"

mkdir -p "${BATCH_DIR}"
printf "algorithm\tgame\tseed\treturn_mean\treturn_std\treturn_min\treturn_max\tlog\n" \
  > "${BATCH_RESULTS}"

{
  echo "Running ${ALGO} on Qbert and BattleZone with W&B"
  echo "GPU: ${DEVICE}"
  echo "Seeds: ${START_SEED}-${END_SEED}"
  echo "Batch directory: ${BATCH_DIR}"
  echo

  for game in qbert battlezone; do
    for seed in $(seq "${START_SEED}" "${END_SEED}"); do
      echo "=== ${ALGO} | ${game} | seed ${seed} ==="
      bash "${SCRIPT_DIR}/run_atari100k_wandb.sh" \
        "${ALGO}" "${game}" "${DEVICE}" "${seed}" "${BATCH_DIR}"

      RUN_LOG="${BATCH_DIR}/atari100k_${ALGO}_${game}_seed${seed}/train_eval.log"
      RETURN_MEAN="$(awk '/eval\/return_mean:/ {print $2}' "${RUN_LOG}" | tail -n 1)"
      RETURN_STD="$(awk '/eval\/return_std:/ {print $2}' "${RUN_LOG}" | tail -n 1)"
      RETURN_MIN="$(awk '/eval\/return_min:/ {print $2}' "${RUN_LOG}" | tail -n 1)"
      RETURN_MAX="$(awk '/eval\/return_max:/ {print $2}' "${RUN_LOG}" | tail -n 1)"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${ALGO}" "${game}" "${seed}" \
        "${RETURN_MEAN:-NA}" "${RETURN_STD:-NA}" \
        "${RETURN_MIN:-NA}" "${RETURN_MAX:-NA}" "${RUN_LOG}" \
        >> "${BATCH_RESULTS}"
      echo
    done
  done

  echo "Finished all 10 runs."
} 2>&1 | tee -a "${BATCH_LOG}"
