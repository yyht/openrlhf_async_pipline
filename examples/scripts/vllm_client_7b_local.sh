#!/bin/bash

set -x
pids=()

export MODEL_PATH=/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/
CUDA_VISIBLE_DEVICES='0' python3 -m openrlhf.async_pipline.vllm_serve \
    --model ${MODEL_PATH} \
    --tensor_parallel_size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 16384 &
pids+=($!)

CUDA_VISIBLE_DEVICES='1' python3 -m openrlhf.async_pipline.vllm_serve \
    --model ${MODEL_PATH} \
    --tensor_parallel_size 1 \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 16384 &
pids+=($!)

CUDA_VISIBLE_DEVICES='2' python3 -m openrlhf.async_pipline.vllm_serve \
    --model ${MODEL_PATH} \
    --tensor_parallel_size 1 \
    --host 0.0.0.0 \
    --port 8002 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 16384 &
pids+=($!)

CUDA_VISIBLE_DEVICES='3' python3 -m openrlhf.async_pipline.vllm_serve \
    --model ${MODEL_PATH} \
    --tensor_parallel_size 1 \
    --host 0.0.0.0 \
    --port 8003 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 16384 &
pids+=($!)

for pid in ${pids[*]}; do
    wait $pid
done

sleep 100d;

