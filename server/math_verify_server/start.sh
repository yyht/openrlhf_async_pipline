

#!/bin/bash

apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/*

apt-get update && apt-get -y install sudo

N_GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}
echo "Number of GPUS: $N_GPUS"
NUM_PROCESSES=$(expr $N_GPUS \* $WORLD_SIZE)
echo "Number of process: $NUM_PROCESSES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

export IP_ADDRESS=$(ifconfig eth1 | grep 'inet ' | awk '{print $2}')
export PORT=$((3000 + RANK))

# if [ "$RANK" -eq 0 ]; then
#     rm -r /cpfs/user/chenhao/ppo_rule_gym/math/rule_rm_url/*
#     mkdir /cpfs/user/chenhao/ppo_rule_gym/math/rule_rm_url/
# fi

export YOUR_PATH=xxx

export ROOT_PATH=${YOUR_PATH}/math_verify_gunicorn_cpu_fix_latest_v3/python/
export LOG_PATH=${YOUR_PATH}/python_log/
export LOG_DIR=${YOUR_PATH}/math_verify_log_cpu_fix_latest_v3
export GC_INTERVAL='43200'

mkdir ${ROOT_PATH}
mkdir ${LOG_PATH}

set -x
if [ "$RANK" -eq 0 ]; then
    mkdir ${ROOT_PATH}
    rm -r ${ROOT_PATH}*
    mkdir ${LOG_PATH}
    rm -r ${LOG_PATH}*
    rm -r ${LOG_DIR}
    rm -r ${LOG_DIR}*
else
    sleep 30s;
fi

export ADDRESS_FILE=${ROOT_PATH}'qwen25_math_deploy_summary_'${RANK}'.txt'
export TEMP_FOLDER=${LOG_PATH}'sample_info_'${RANK}'.jsonl'

pip3 uninstall latex2sympy2 -y

pip3 install math-verify jsonlines coloredlogs pysnooper Fraction ema_pytorch --upgrade
pip3 install apscheduler gunicorn transformers==4.45.2 trl==0.9.6 pysbd jsonlines coloredlogs pebble -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com
pip3 install func_timeout sentencex requests_futures timeout_decorator word2number math-verify flashtext pygments -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

# # Set a while loop and watch port 
# PORT_ONLINE=$(sudo netstat -ntlp | grep :${PORT})
# while [ -z "${PORT_ONLINE}" ]
# do
#     sleep 1
#     PORT_ONLINE=$(sudo netstat -ntlp | grep :${PORT})
# done

echo "IP_ADDRESS: $IP_ADDRESS"
echo "PORT: $PORT"
echo "http://${IP_ADDRESS}:${PORT}" >> $ADDRESS_FILE

bash run_remote_math_verify_dist.sh