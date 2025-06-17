<div align="center">
    <img alt="OpenRLHF logo" src="./docs/logo.png" style="height: 140px;" />
</div>
<div align="center">
<p align="center">
      <a href="https://github.com/OpenRLHF/OpenRLHF/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/OpenRLHF/OpenRLHF" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/OpenRLHF/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/OpenRLHF/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/OpenRLHF/OpenRLHF?color=0088ff" />
      <a href="https://github.com/OpenRLHF/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/OpenRLHF/OpenRLHF?color=ccf" />
      </a>
      <br>
      <em>Open-source / Comprehensive / Lightweight / Easy-to-use</em>
    </p>
</p>
</div>

<hr>

<span>[ English | <a href="README_zh.md">中文</a> | <a href="README_ja.md">日本語</a> ]</span>

OpenRLHF is a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers:

- **Simple and easy to use**: OpenRLHF is one of the simplest high-performance RLHF libraries currently available, and seamlessly compatible with Huggingface models and datasets.
- **High performance**: RLHF training spends 80% of the time on the sample generation stage. Thanks to the ability to use a large inference batch size with Ray and Packing Samples and vLLM generation acceleration, the performance of OpenRLHF 3~4x+ that of Optimized DeepSpeedChat with Hybrid Engine.
- **Distributed RLHF**:  OpenRLHF distribute the Actor, Reward, Reference, and Critic models onto separate GPUs using Ray, while placing the Adam optimizer on the CPU. This enables full-scale fine-tuning of 70B+ models with multiple A100 80G GPUs and vLLM and 7B models across multiple 24GB RTX 4090 GPUs.
- **Hybrid Engine**:  OpenRLHF also supports the hybrid engine, allowing all models and vLLM engines to share the GPUs to avoid GPU idling.
- **PPO Implementation Optimization**: We integrated the implementation tricks for PPO to improve the training stability, referencing [Zhihu](https://zhuanlan.zhihu.com/p/622134699) and [Advanced Tricks for Training Large Language Models with Proximal Policy Optimization](https://hijkzzz.notion.site/rlhf-implementation-tricks?v=158d9a33ecc98132bf9e000c39227361).

## interactions-scaling

![scaling_img](docs/interactions_scaling.png)
The model is released at [huggingface](https://huggingface.co/htxu91/zero-tir-7b-550step).
The evaluation script is located at ./evaluation. You should be careful to set your openrlhf-path in ./evaluation/my_evaluation.py


## Features

- ASYNC-PIPLINE [PPO](./examples/scripts/train_long_cot_ray_7b_reinforce_baseline_tune_zero_v1_onpolicy_env_mask_async_async.sh). We have test async-rollout with pipline to incentivize the reasoning ability on math-tasks with multiturn TIR(tool-intergated-reasoning). It has been tested on 7b/32b for reinforce++ and grpo with env-mask to exclude the loss calculation for env-feedback. [Zhihu](https://zhuanlan.zhihu.com/p/1903425641954674326). It achieves better performance on AIME24/25 with fewer training-steps compared to zeor-rl-text-cot.
- DETAIL:
    - ./trainer/ray/async_vllm_engine_async.py(function **def add_env_pipline_requests**) for async-rollout.
    - In order to unify the multi-turn tool rl, we further add ./openrlhf/async_pipline/process_request.py to handle different env with the **env_func** from label(suppose label is constructed via json.dumps({})). So, you just need to add different env_func as showed in /env/math/math_tir_process_single_request.py(tir) or default generation method showed in ./openrlhf/async_pipline/process_request.py(function: **def default_generate**)
    - The make_experience.py decouples the rollout-generation and make-exp(calculate the advantages/logprob and so on.). You should focus on function **run_async_queue/gather_queue/put_queue** in ./trainer/ppo_utils/experience_maker.py
    - The ppo-trainer.py also modified to support pipline. ./trainer/ppo_trainer.py(See function **fit**).
- RUNNING SCRIPT
```
bash 
export WARMUP=0.0
export LR=1e-6
export KL=0.0
export ENTROPY_RATIO=0.0
export STRUCTURED_REWARD="STRUCTURED_REWARD"
export GENERATE_METHOD='math_tir_generate'
export DEBUG_PATH='/cpfs/user/debug/tir'
export VLLM_USE_V1=0
export USE_MODEL_REWARD='yes'
export MAX_CONCURRENT=128
export NCCL_TIMEOUT=36000000

mkdir ${DEBUG_PATH}

export USE_REMOTE_RM_ACTOR="RayActor"
export CONCURRENCY='1'
export ENV_ITER_NUM='2'
 
export expname=xxx/qwen25_32B_reinforce_baseline_zero_tir_fix_boxed_lr${LR}_warmup${WARMUP}_kl${KL}_zero_tir_0506_nginx_prefetch_fix_env_mask_vllm083_xverify_deepmath_async_iternum${ENV_ITER_NUM}/
export SAVE_PATH=xxx/qwen25_32B_reinforce_baseline_zero_tir_fix_boxed_lr${LR}_warmup${WARMUP}_kl${KL}_zero_tir_0506_nginx_prefetch_fix_env_mask_vllm083_xverify_deepmath_async_iternum${ENV_ITER_NUM}/

rm -r ${expname}roll*

export PRETRAIN=xxx/Qwen/Qwen2.5-32B-local/
export REF_PRETRAIN=xxx/Qwen/Qwen2.5-32B-local/
export REWARD_PRETRAIN=xxx/Qwen/Qwen2.5-32B-local/

export PROMPT_DATA=xxx/DeepMath-103K/deepmath_103k.jsonl
export PROMPT_DATA_PROBS='1.0'

export TENSORBOARD=${SAVE_PATH}/tensorboard/
export REMOTE_RM_URL='http://xxx:yyy' # remote-reward-url
export COMPILE_SERVER='http://xxx:yyy' # remote-python-compile

mkdir ${SAVE_PATH}
mkdir ${TENSORBOARD}
mkdir ${expname}

export PATH=$HOME/.local/bin/:$PATH
export TEMPLATE_NAME='ZERO_TIR'

set -x
if [ "$RANK" -eq 0 ]; then
    ray start --head --port=6379  --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus 8 
    ifconfig net0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1 > $expname/node_ip.txt
    export MASTER_NODE=$(cat $expname/node_ip.txt)
    sleep 2m
    ./train_long_cot_ray_32b_reinforce_baseline_tune_zero_v1_onpolicy_env_mask_async_async_pipline_continue.sh
else
    sleep 1m
    export MASTER_NODE=$(cat $expname/node_ip.txt)
    ray start --address="${MASTER_NODE}:6379"
fi
 
sleep 365d
```

## Companies and Organizations using OpenRLHF

- Google
- ByteDance
- Tencent
- Alibaba
- Baidu
- China Telecom
- Vivo
- Allen AI
- NexusFlow
- Jülich Supercomputing Centre (JSC)
- Berkeley Starling Team
- M-A-P
- ...

## Join Us

**How to Join?**

1. Email us at janhu9527@gmail.com or join [GitHub Organization](https://github.com/OpenRLHF). Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenRLHF ↗](https://github.com/OpenRLHF/OpenRLHF) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of the OpenRLHF project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Sponsor Us

Your sponsorship can help us maintain and improve OpenRLHF. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/OpenRLHF).

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=OpenRLHF/OpenRLHF&type=Date)](https://star-history.com/#OpenRLHF/OpenRLHF&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/OpenRLHF/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenRLHF/OpenRLHF" />
</a>

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA ↗](https://llama.meta.com/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

Our project would also like to thank [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) and [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat). In the early stages of the project, we referred to their code design. 
Our project would like to thank [Netmind.AI](https://www.netmind.ai/) for the GPU support of developing ring attention.

(2024/7) Our GitHub organization has changed from OpenLLMAI to OpenRLHF.

## Citation
```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Zilin Zhu and Xianyu and Weixun Wang and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```

______________________________________________________________________

*OpenRLHF © 2025 OpenRLHF. All Rights Reserved.*
