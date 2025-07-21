
# apt-get update && \
#     apt-get install -y gosu && \
#     rm -rf /var/lib/apt/lists/*

# apt-get update && apt-get -y install sudo

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_SHM_DISABLE=1
# export NCCL_DEBUG=INFO

cd /cpfs/user/chenhao/debug/OpenRLHF_082/
# git clone https://github.com/OpenRLHF/OpenRLHF.git
pip3 install -e . -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

pip3 install deepspeed==0.17.0

# cd /cpfs/user/chenhao/debug/
# cp nccl.conf /etc/nccl.conf
# echo "COPY nccl.conf to etc"
# cp parameter_offload.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/parameter_offload.py
# echo "COPY parameter_offload to deepspeed"
# cp partitioned_param_coordinator.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py
# echo "COPY partitioned_param_coordinator to deepspeed"

pip3 install math-verify loguru fastapi uvicorn httpx python-multipart aiohttp aiolimiter pysbd jsonlines coloredlogs pebble aiolimiter -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com
pip3 install func_timeout sentencex requests_futures timeout_decorator flashtext pygments -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

cd /cpfs/user/chenhao/debug/OpenRLHF_082/
chmod -R 777 ./examples/scripts/

ln -s /cpfs/user/chenhao/debug/OpenRLHF_082/ /openrlhf
cd /openrlhf/examples/scripts
chmod -R 777 /openrlhf/examples/scripts

# export PRETRAIN_PATH=/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/
# export SAVE_PATH=/cpfs/user/chenhao/outputs/Llama-3-8b-sft-mixture-agent/
# export DATA_PATH=/cpfs/user/chenhao/hf_datasets/prompt-collection-v0.1

export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export USE_MODEL_REWARD='yes'
export NCCL_TIMEOUT=36000000
export MAX_CONCURRENT=512
export TASK_MAX_CONCURRENT=32
export MAX_VLLM_BATCHSIZE=4
export ENV_ITER_NUM='2'
export WARMUP=0.0
export LR=1e-6
export KL=0.0
export OPENRLHF_PATH=/cpfs/user/chenhao/debug/OpenRLHF_082/
export OPENRLHF_ASYNC_QUEUE_SIZE=1
export N_ROLLOUT=16
export MAX_COMPILE_RETRIES=2
export OPENRLHF_ASYNC_QUEUE_SIZE=1
# export QUALITY_REWARD=0.1
# export OVER_LONG_REWARD=0.1
export CODE_CONCURRENT=2
# export USE_SHORTCUT_REWARD=0.1
# export USE_FORMAT_REWARD='yes'
# export USE_TIR_FORMAT_LOOSE=0.1

mkdir /newcpfs/user/chenhao/outputs/

export SAVE_PATH=/newcpfs/user/chenhao/outputs/qwen25_7B_rloo_zero_tir_lr${LR}_warmup${WARMUP}_kl${KL}_zero_0707_agent_tir_iternum${ENV_ITER_NUM}_queue_size${OPENRLHF_ASYNC_QUEUE_SIZE}_token_level_rolloutn${N_ROLLOUT}_orz_dapo_seqbalance_raw_adamw_before_select_reuse_dualclip_lossmask/
export expname=${SAVE_PATH}

export PRETRAIN=/newcpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/
export REF_PRETRAIN=/newcpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/
export DATA_PATH=/cpfs/user/chenhao/Open-Reasoner-Zero/data/level_difficulity_problem.tir.jsonl,/cpfs/user/chenhao/hf_datasets/DAPO-Math-17k/dapo.tir.jsonl

export TENSORBOARD=${SAVE_PATH}/tensorboard/
export NGINX_IP_FILE=/cpfs/user/chenhao/hf_datasets/qwen25_qwq/nginx_conf/nginx_ip.txt
export COMPILE_SERVER_PORT='10003'
export MATH_VERIFY_SERVER_PORT='10008'
export XVERIFY_MATH_MODEL_SERVER_PORT='10005'
export REMOTE_RM_URL='http://10.39.2.54:10007'

mkdir ${SAVE_PATH}
mkdir ${TENSORBOARD}
mkdir ${expname}

export PATH=$HOME/.local/bin/:$PATH

set -x
if [ "$RANK" -eq 0 ]; then
    ray start --head --port=6379  --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus 8
    ifconfig net0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1 > $expname/node_ip.txt
    export MASTER_NODE=$(cat $expname/node_ip.txt)
    sleep 2m
    set -x
    ray job submit --address="http://${MASTER_NODE}:8265/" \
        --runtime-env-json='{"working_dir": "/openrlhf"}' \
        -- python3 -m openrlhf.cli.train_ppo_ray \
        --ref_num_nodes 1 \
        --ref_num_gpus_per_node 8 \
        --actor_num_nodes 1 \
        --actor_num_gpus_per_node 8 \
        --vllm_num_engines 8 \
        --vllm_tensor_parallel_size 1 \
        --colocate_actor_ref \
        --vllm_gpu_memory_utilization 0.9 \
        --gamma 1.0 \
        --l2 0.01 \
        --async_train \
        --dynamic_filtering \
        --dynamic_filtering_reward_range 0.1 0.8 \
        --eps_clip_low_high 0.2 0.28 \
        --advantage_estimator rloo \
        --pretrain ${PRETRAIN} \
        --ref_pretrain ${REF_PRETRAIN} \
        --agent_func_path /openrlhf/examples/python/agent_func.py \
        --save_path ${SAVE_PATH} \
        --ckpt_path ${SAVE_PATH} \
        --save_hf_ckpt \
        --micro_train_batch_size 4 \
        --train_batch_size 2048 \
        --micro_rollout_batch_size 4 \
        --rollout_batch_size 128 \
        --repeatness_threshold 0.05 \
        --n_samples_per_prompt ${N_ROLLOUT} \
        --max_epochs 1 \
        --num_episodes 100000000 \
        --prompt_max_len 1024 \
        --max_samples 100000000 \
        --generate_max_len 8192 \
        --zero_stage 3 \
        --bf16 \
        --init_kl_coef ${KL} \
        --lr_warmup_ratio ${WARMUP} \
        --actor_learning_rate ${LR} \
        --critic_learning_rate 9e-6 \
        --prompt_data ${DATA_PATH} \
        --input_key query \
        --label_key label \
        --normalize_reward \
        --gradient_checkpointing \
        --use_global_token_level_loss \
        --select_before_normalize \
        --reuse_before_normalize \
        --remove_advantage_whiten \
        --use_dual_policy_loss \
        --use_loss_mask \
        --filter_sample \
        --vllm_sync_backend nccl \
        --vllm_enable_sleep \
        --deepspeed_enable_sleep \
        --adam_offload \
        --flash_attn \
        --normalize_reward \
        --gradient_checkpointing \
        --packing_samples \
        --enforce_eager \
        --load_checkpoint \
        --save_steps 50 \
        --use_tensorboard ${TENSORBOARD} \
        --remote_rm_url ${REMOTE_RM_URL}
else
    sleep 1m
    export MASTER_NODE=$(cat $expname/node_ip.txt)
    ray start --address="${MASTER_NODE}:6379"
fi
 
sleep 365d