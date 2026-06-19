#!/usr/bin/env bash
set -euo pipefail

ALGO="${1:-der}"
GAME="${2:-qbert}"
DEVICE="${3:-2}"
SEED="${4:-5000}"

cd "$(dirname "$0")/.."
RUN_ROOT="${5:-${PWD}/logs}"

case "${ALGO}" in
  der|spr|sr_spr|bbf|sac_bbf) ;;
  *)
    echo "Unsupported algorithm: ${ALGO}" >&2
    echo "Usage: $0 [der|spr|sr_spr|bbf|sac_bbf] [qbert|battlezone] [gpu_id] [seed] [run_root]" >&2
    exit 2
    ;;
esac

case "${GAME}" in
  qbert|battlezone) ;;
  *)
    echo "Unsupported game: ${GAME}" >&2
    echo "Usage: $0 [der|spr|sr_spr|bbf|sac_bbf] [qbert|battlezone] [gpu_id] [seed] [run_root]" >&2
    exit 2
    ;;
esac

RUN_NAME="atari100k_${ALGO}_${GAME}_seed${SEED}"
OUT_DIR="${RUN_ROOT}/${RUN_NAME}"
LOG_FILE="${OUT_DIR}/train_eval.log"

mkdir -p "${OUT_DIR}"
: > "${LOG_FILE}"

{
  echo "Running ${ALGO} on ${GAME}: seed=${SEED}, gpu=${DEVICE}, logger=wandb"
  echo "Output directory: ${OUT_DIR}"
  echo "Started at: $(date --iso-8601=seconds)"
  echo

  .venv/bin/python src/train.py "experiment=atari100k/${ALGO}/${GAME}" \
    "trainer.accelerator=gpu" \
    "trainer.devices=[${DEVICE}]" \
    "trainer.seed=${SEED}" \
    "logger=[wandb]" \
    "checkpoint.save_dir=${OUT_DIR}/checkpoints" \
    "hydra.run.dir=${OUT_DIR}"

  echo
  echo "Finished at: $(date --iso-8601=seconds)"
} 2>&1 | tee -a "${LOG_FILE}"
