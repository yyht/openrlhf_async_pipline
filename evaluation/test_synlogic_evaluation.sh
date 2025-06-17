
pip3 install math-verify loguru fastapi uvicorn httpx python-multipart aiohttp aiolimiter pysbd jsonlines coloredlogs pebble aiolimiter -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com
pip3 install func_timeout sentencex requests_futures timeout_decorator flashtext pygments -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com


export MODEL_NAME_OR_PATH=/newcpfs/user/chenhao/outputs/qwen25_7B_reinforce_baseline_zero_tir_lr1e-6_warmup0.0_kl0.0_zero_0526_latest_agent_iternum2_synlogic_easy_rollout16/global_step100_hf/
export DATA_NAME=synlogic.easy.val.jsonl
export DATA_DIR=/newcpfs/user/chenhao/hf_datasets/DATA_DIR/
export OUTPUT_DIR=/newcpfs/user/chenhao/outputs/qwen25_7B_reinforce_baseline_zero_tir_lr1e-6_warmup0.0_kl0.0_zero_0526_latest_agent_iternum2_synlogic_easy_rollout16/global_step100_hf/
export PROMPT_TYPE=synlogic
export INPUT_KEY=query
export ANSWER_KEY=label
export TEMPERATURE=0.0
export N_SAMPLING=1
export top_p=1.0

sh synlogic_evaluation.sh