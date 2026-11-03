#!/bin/bash
# Runs rabitq_bench with the RAPIDS libraries shipped in the project venv.
SP=/home/cc/hpdic/certex/.venv/lib/python3.12/site-packages
SP=$(cd "$SP" && pwd)
export LD_LIBRARY_PATH=$SP/libcuvs/lib64:$SP/librmm/lib64:$SP/rapids_logger/lib64:$SP/libraft/lib64:$SP/libkvikio/lib64:$SP/nvidia/nvjitlink/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cusolver/lib:$SP/nvidia/cusparse/lib:$SP/nvidia/curand/lib:$SP/nvidia/nccl/lib:$SP/nvidia/cuda_runtime/lib:$SP/libcuvs_cu12.libs:$LD_LIBRARY_PATH
exec "$(dirname "$0")/rabitq_bench" "$@"
