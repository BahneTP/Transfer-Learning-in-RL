#!/usr/bin/env bash
set -euo pipefail

ALGO="${1:-der}"
GAME="${2:-qbert}"
DEVICE="${3:-2}"
SEED="${4:-5000}"

cd "$(dirname "$0")/.."

case "${ALGO}" in
  der|spr|bbf) ;;
  *)
    echo "Unsupported algorithm: ${ALGO}" >&2
    echo "Usage: $0 [der|spr|bbf] [qbert|battlezone] [gpu_id] [seed]" >&2
    exit 2
    ;;
esac

case "${GAME}" in
  qbert|battlezone) ;;
  *)
    echo "Unsupported game: ${GAME}" >&2
    echo "Usage: $0 [der|spr|bbf] [qbert|battlezone] [gpu_id] [seed]" >&2
    exit 2
    ;;
esac

RUN_NAME="atari100k_${ALGO}_${GAME}_seed${SEED}"
OUT_DIR="${PWD}/logs/${RUN_NAME}"
LOG_FILE="${OUT_DIR}/train_eval.log"

mkdir -p "${OUT_DIR}"
: > "${LOG_FILE}"

{
  echo "Running ${ALGO} on ${GAME}: seed=${SEED}, gpu=${DEVICE}"
  echo "Output directory: ${OUT_DIR}"
  echo "Started at: $(date --iso-8601=seconds)"
  echo

  .venv/bin/python src/train.py "experiment=atari100k/${ALGO}/${GAME}" \
    "trainer.accelerator=gpu" \
    "trainer.devices=[${DEVICE}]" \
    "trainer.seed=${SEED}" \
    "logger=[]" \
    "checkpoint.save_dir=${OUT_DIR}/checkpoints" \
    "hydra.run.dir=${OUT_DIR}"

  echo
  echo "Starting evaluation at: $(date --iso-8601=seconds)"
  echo

  .venv/bin/python src/eval.py "experiment=atari100k/${ALGO}/${GAME}" \
    "trainer.accelerator=gpu" \
    "trainer.devices=[${DEVICE}]" \
    "trainer.seed=${SEED}" \
    "trainer.num_eval_episodes=100" \
    "checkpoint.resume_from=${OUT_DIR}/checkpoints/last.pt" \
    "hydra.run.dir=${OUT_DIR}/eval"

  echo
  echo "Finished at: $(date --iso-8601=seconds)"
} 2>&1 | tee -a "${LOG_FILE}"
