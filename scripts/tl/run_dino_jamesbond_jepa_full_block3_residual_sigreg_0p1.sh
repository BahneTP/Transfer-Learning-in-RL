#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JEPA_WEIGHT="${JEPA_WEIGHT:-1.0}" \
JEPA_PREDICTION_MODE="${JEPA_PREDICTION_MODE:-residual}" \
LAMBDA_SIGREG="${LAMBDA_SIGREG:-0.1}" \
RUN_LABEL="${RUN_LABEL:-weight_1_residual_sigreg_0p1}" \
  exec "${SCRIPT_DIR}/run_dino_jamesbond_jepa_full_block3.sh" "$@"
