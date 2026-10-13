#!/usr/bin/env bash
# Helper script to run commands in the mixnet environment on v21
# Usage: ./scripts/run_v21.sh python run_MixNet.py ...

export CONDA_ENV="/ist-nas/users/pratchp/conda_envs/mixnet"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
export PATH="${CONDA_ENV}/bin:${PATH}"

# Default to GPU 0 if not specified
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

exec "$@"
