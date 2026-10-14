#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export DYLD_LIBRARY_PATH="$PROJECT_ROOT/.runtime/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv/bin/python" "$@"
