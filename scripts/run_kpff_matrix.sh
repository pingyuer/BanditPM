#!/usr/bin/env bash
set -euo pipefail

METHOD="${METHOD:-kpff}" DATASET="${DATASET:-all}" bash "$(dirname "$0")/run_canonical_matrix.sh" "$@"
