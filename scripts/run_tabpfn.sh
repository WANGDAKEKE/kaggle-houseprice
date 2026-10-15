#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export DYLD_LIBRARY_PATH="$PROJECT_ROOT/.runtime/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export TABPFN_MODEL_CACHE_DIR="$PROJECT_ROOT/.cache/tabpfn/models"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TABPFN_NO_BROWSER=true
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TABPFN_MPS_MEMORY_FRACTION=0.5
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv-tabpfn/bin/python" "$@"
