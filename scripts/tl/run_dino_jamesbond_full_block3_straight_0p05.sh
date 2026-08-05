#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JEPA_WEIGHT="${JEPA_WEIGHT:-0.0}" \
JEPA_PREDICTION_MODE="${JEPA_PREDICTION_MODE:-residual}" \
TEMPORAL_STRAIGHTENING_WEIGHT="${TEMPORAL_STRAIGHTENING_WEIGHT:-0.05}" \
LAMBDA_SIGREG="${LAMBDA_SIGREG:-0.0}" \
RUN_LABEL="${RUN_LABEL:-straight_0p05}" \
  exec "${SCRIPT_DIR}/run_dino_jamesbond_jepa_full_block3.sh" "$@"
