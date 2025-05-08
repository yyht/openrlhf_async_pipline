
#!/bin/bash
N_GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}
echo "Number of GPUS: $N_GPUS"
NUM_PROCESSES=$(expr $N_GPUS \* $WORLD_SIZE)
echo "Number of process: $NUM_PROCESSES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
N_PROCESS=$N_GPUS
MACHINE_NUM=$(expr $WORLD_SIZE \* 8)

echo "N_PROCESS: $N_PROCESS"
echo "MACHINE_NUM: $MACHINE_NUM"

echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"

# 用于存放子进程的PID
pids=()

mkdir ${SAVE_PATH}
echo ${SAVE_PATH} ${MODEL_PATH}
set -x
for LOCAL_RANK in $(seq 0 $((N_PROCESS - 1)))
do
    SPLIT=$(expr $LOCAL_RANK \+ $RANK \* 8)
    PORT=$(expr 10000 \+ $RANK \+ $LOCAL_RANK)

    echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
    
    echo "IP_ADDRESS: $IP_ADDRESS"
    echo "PORT: $PORT"
    echo "http://${IP_ADDRESS}:${PORT}" >> $ADDRESS_FILE

    CUDA_VISIBLE_DEVICES=${LOCAL_RANK} python3 -m openrlhf.async_pipline.vllm_serve \
    --model ${MODEL_PATH} \
    --tensor_parallel_size 1 \
    --host 0.0.0.0 \
    --port ${PORT} \
    --gpu_memory_utilization 0.9 \
    --max_model_len 16384 &
    pids+=($!)
done

for pid in ${pids[*]}; do
    wait $pid
done

sleep 100d;
