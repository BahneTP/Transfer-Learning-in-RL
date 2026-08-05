#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JEPA_WEIGHT="${JEPA_WEIGHT:-0.01}" RUN_LABEL="${RUN_LABEL:-weight_0p01}" \
  exec "${SCRIPT_DIR}/run_dino_jamesbond_jepa_full_block3.sh" "$@"
