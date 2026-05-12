#!/usr/bin/env bash
set -euo pipefail

METHOD="${METHOD:-unext_fusion}" DATASET="${DATASET:-all}" bash "$(dirname "$0")/run_canonical_matrix.sh" "$@"
