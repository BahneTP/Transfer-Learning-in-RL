#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <gpu_id> <experiment>" >&2
  echo "Example: $0 3 rainbow/qbert_atari100k" >&2
  exit 2
fi

GPU_ID="$1"
EXPERIMENT="$2"
GAME="${EXPERIMENT##*/}"
GAME="${GAME%_atari100k}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

RUN_NAME="$(echo "${EXPERIMENT}" | tr '/' '_')_gpu${GPU_ID}_$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/logs/atari100k_single/${RUN_NAME}}"
TRAIN_LOG="${RUN_DIR}/train.log"
EVAL_LOG="${RUN_DIR}/eval.log"
CHECKPOINT="${RUN_DIR}/train/checkpoints/last.pt"

mkdir -p "${RUN_DIR}"

echo "Training experiment=${EXPERIMENT} on GPU ${GPU_ID}"
echo "Logs: ${RUN_DIR}"

python -u src/train.py \
  "experiment=${EXPERIMENT}" \
  "trainer.accelerator=gpu" \
  "trainer.devices=[${GPU_ID}]" \
  "logger=[]" \
  "checkpoint.save_every_n_steps=999999999" \
  "checkpoint.save_last=true" \
  "hydra.run.dir=${RUN_DIR}/train" \
  2>&1 | tee "${TRAIN_LOG}"

echo
echo "Evaluating checkpoint: ${CHECKPOINT}"

python -u src/eval.py \
  "algorithm=rainbow_atari100k" \
  "environment=${GAME}_atari100k_train" \
  "environment@eval_environment=${GAME}_atari100k_eval" \
  "trainer.accelerator=gpu" \
  "trainer.devices=[${GPU_ID}]" \
  "logger=[]" \
  "checkpoint.resume_from=${CHECKPOINT}" \
  "hydra.run.dir=${RUN_DIR}/eval" \
  2>&1 | tee "${EVAL_LOG}"

echo
echo "Final evaluation results:"
awk '/Evaluation results:/,0 {print}' "${EVAL_LOG}"
