


#!/bin/bash

N_GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}
echo "Number of GPUS: $N_GPUS"
NUM_PROCESSES=$(expr $N_GPUS \* $WORLD_SIZE)
echo "Number of process: $NUM_PROCESSES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

N_PROCESS=$(expr $COLOCATE \* $N_GPUS \* $WORLD_SIZE)
MACHINE_NUM=$(expr $WORLD_SIZE \* $N_GPUS)

echo "N_PROCESS: $N_PROCESS"
echo "MACHINE_NUM: $MACHINE_NUM"

export VLLM_WORKER_MULTIPROC_METHOD='spawn'

# 用于存放子进程的PID
pids=()

# MODEL_PATH=/data/nas/chenhao/pretrained_models/Qwen/Qwen2.5-Math-7B-Instruct
# SAVE_PATH=/data/nas/chenhao/hf_datasets/qwen25_math_pdf/

set -x
for LOCAL_RANK in $(seq 0 $((N_GPUS - 1)))
do
    for COLOCATE_RANK in $(seq 1 $((COLOCATE)))
    do
        PORT=$(expr $COLOCATE_RANK \+ $LOCAL_RANK \* $COLOCATE \+ $RANK \* $N_GPUS \+ $INIT_PORT)
        echo ${COLOCATE_RANK} ${PORT}
        echo "http://${IP_ADDRESS}:${PORT}" >> $ADDRESS_FILE
        CUDA_VISIBLE_DEVICES=${LOCAL_RANK} python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH}  \
        --tp 1 \
        --nnodes 1 \
        --node-rank 0 \
        --trust-remote-code \
        --chunked-prefill-size 2048 \
        --host 0.0.0.0 \
        --port $PORT \
        --watchdog-timeout 36000 \
        --max-running-requests 1000 \
        --schedule-conservativeness 1.2 \
        --max-total-tokens 8192 \
        --enable-torch-compile \
        --mem-fraction-static 0.5 \
        --disable-cuda-graph &
        pids+=($!)
    done
done

for pid in ${pids[*]}; do
    wait $pid
done

echo "All processes have completed."


