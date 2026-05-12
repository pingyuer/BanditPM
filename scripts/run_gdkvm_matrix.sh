#!/usr/bin/env bash
set -euo pipefail

METHOD="${METHOD:-gdkvm}" DATASET="${DATASET:-all}" bash "$(dirname "$0")/run_canonical_matrix.sh" "$@"
