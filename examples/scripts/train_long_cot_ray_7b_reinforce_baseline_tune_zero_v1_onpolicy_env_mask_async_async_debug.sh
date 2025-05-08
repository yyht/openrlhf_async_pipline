set -x 

# ray job submit --address="http://127.0.0.1:8265" \
ray job submit --address="http://${MASTER_NODE}:8265/" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray_async_async_server \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain ${PRETRAIN} \
   --critic_pretrain ${PRETRAIN} \
   --save_path ${SAVE_PATH} \
   --ckpt_path ${SAVE_PATH} \
   --micro_train_batch_size 1 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 64 \
   --max_samples 10000000000 \
   --max_epochs 1 \
   --num_episodes 1000000 \
   --prompt_max_len 1024 \
   --generate_max_len 16000 \
   --zero_stage 3 \
   --save_steps 5 \
   --use_prefetch \
   --use_env_mask \
   --use_acc_filter \
   --reuse_offline \
   --dump_data \
   --use_eval_filter \
   --use_length_filter \
   --freezing_actor_steps -1 \
   --n_samples_per_prompt 2 \
   --entropy_loss_coef ${ENTROPY_RATIO} \
   --advantage_estimator 'reinforce_baseline' \
   --bf16 \
   --lr_warmup_ratio ${WARMUP} \
   --actor_learning_rate ${LR} \
   --critic_learning_rate 9e-6 \
   --init_kl_coef ${KL} \
   --max_ckpt_num 100 \
   --prompt_data ${PROMPT_DATA} \
   --prompt_data_probs ${PROMPT_DATA_PROBS} \
   --input_key query \
   --label_key label \
   --adam_offload \
   --flash_attn \
   --normalize_reward \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_tensorboard ${TENSORBOARD} \
   --remote_rm_url ${REMOTE_RM_URL}


# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)