# #!/bin/bash


# apt-get update && \
#     apt-get install -y gosu && \
#     rm -rf /var/lib/apt/lists/*

# apt-get update && apt-get -y install sudo

export IP_ADDRESS=$(hostname -I | awk '{print $1}')

N_GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}
echo "Number of GPUS: $N_GPUS"
NUM_PROCESSES=$(expr $N_GPUS \* $WORLD_SIZE)
echo "Number of process: $NUM_PROCESSES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

export YOUR_PATH=xxx # /root/xverify/log

export ADDRESS_FILE=${YOUR_PATH}'_'${RANK}'.txt'
> $ADDRESS_FILE

pip3 install jsonlines coloredlogs pysnooper Fraction ema_pytorch --upgrade -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com
pip3 install trl==0.9.6 pysbd jsonlines coloredlogs pebble -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com
pip3 install func_timeout sentencex requests_futures timeout_decorator latex2sympy2 word2number -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

export PATH=$HOME/.local/bin/:$PATH

export COLOCATE=4
export MODEL_PATH=xVerify-3B-Ia/ # xverify-model-path

export INIT_PORT=12000
bash deploy_xverify_3b_sglang.sh