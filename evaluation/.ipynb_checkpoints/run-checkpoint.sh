
# apt-get update && \
#     apt-get install -y gosu && \
#     rm -rf /var/lib/apt/lists/*

# apt-get update && apt-get -y install sudo

echo "Number of GPUS: $N_GPUS"
echo "Number of process: $NUM_PROCESSES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# export VLLM_PATH=/cpfs/user/chenhao/vllm
# export PYTHONPATH=$VLLM_PATH:$PYTHONPATH

export RANK=${RANK}
export MY_RANK=2
export NUM_PROCESSES=$(expr $RANK \* $MY_RANK)
echo "MY_RANK: $MY_RANK"
echo "RANK: $RANK"
echo "NUM_PROCESSES: $NUM_PROCESSES"
# export VLLM_USE_V1=0

# pip3 install deepspeed==0.16.0

# cd /cpfs/user/chenhao/debug/
# cp nccl.conf /etc/nccl.conf
# echo "COPY nccl.conf to etc"
# cp parameter_offload.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/parameter_offload.py
# echo "COPY parameter_offload to deepspeed"
# cp partitioned_param_coordinator.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py
# echo "COPY partitioned_param_coordinator to deepspeed"

pip3 install math-verify tabulate markdown pysbd jsonlines coloredlogs func_timeout timeout-decorator word2number Pebble -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

pip3 install loguru fastapi uvicorn httpx python-multipart aiohttp aiolimiter pysbd jsonlines coloredlogs pebble aiolimiter -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com
pip3 install func_timeout sentencex requests_futures timeout_decorator flashtext pygments -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

pip3 install math-verify loguru fastapi uvicorn httpx python-multipart aiohttp aiolimiter pysbd jsonlines coloredlogs pebble aiolimiter -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com
pip3 install func_timeout sentencex requests_futures timeout_decorator flashtext pygments -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

export ROOT_PATH=

export NGINX_IP_FILE=nginx_ip.txt
export COMPILE_SERVER_PORT='10003'
export MATH_VERIFY_SERVER_PORT='10008'
export XVERIFY_MATH_MODEL_SERVER_PORT='10005'
export REMOTE_RM_URL='http://10.39.2.54:10007'
export OPENRLHF_PATH=
export PRETRAIN=

export DEBUG_FLAG='yes'
export CUDA_VISIBLE_DEVICES="0"
export INPUT_KEY='problem,question'
export ANSWER_KEY='answer,final_answer'
export DATA_NAME="aime25,aime24,hmmt_feb_2025,hmmt_feb_2024,cmimc"
export N_SAMPLING=32
export TEMPERATURE=1.0
# export VLLM_USE_V1='0'
export USE_TIR='yes'
export TASK_MAX_CONCURRENT=32

export VLLM_VERSION='vllm085'
export USE_SEPERATE='no'
export USE_ID='USE_ID'

for step in 1100 1000 950 900 850
do
    for iter in 1 2 4 8 16 18 20
    do
        export ENV_ITER_NUM=${iter}
        export MODEL_NAME_OR_PATH=${ROOT_PATH}global_step${step}_hf_actor/
        export OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval_useid
        export PROMPT_TYPE='orz_tir'
        export USE_SEPERATE='yes'
        bash run_evaluation.sh
    done
done