
TEMPLATE = {
    'orz_tir': 
    """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. In your reasoning-process, You can use python-code to solve your problem. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:{input}\nAssistant: <think>""",
    "orz_tir_xinji": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You can use Python code during the solution process, and the code will be executed immediately and the result will be returned. You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:{input}\nAssistant: <think>""",
    "orz_xinji": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:{input}\nAssistant: <think>""",
    "orz_ch": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:{input}\nAssistant: <think>""",
    "qwen25-math-cot-tora": """<|im_start|>system\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n""",
    "deepseek_r1_distill": """<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You should think step-by-step.<｜User｜>{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜>""",
    'synlogic': "{input}\nAssistant: <think>",
    'synlogic_ch': "{input}"
}

import sys, os, asyncio, re
sys.path.append(os.getenv('SYNLOGIC_PATH', '/cpfs/user/chenhao/SynLogic'))
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

from task2verifier import verifier_classes

import random
import os, sys
from timeout_decorator import timeout
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
import openai
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import json, os
import os, sys, uuid

ENV_ITER_NUM = int(os.getenv('ENV_ITER_NUM', '2'))
VLLM_VERSION = os.getenv('VLLM_VERSION', 'vllm_083')

sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_0304_vllm083'))
from env.math.math_tir import math_tir_generate
from passk_eval import estimate_pass_at_k
from tabulate import tabulate

# XVERIFY_MATH_MODEL_SERVER = os.environ.get('XVERIFY_MATH_MODEL_SERVER', None)
# if XVERIFY_MATH_MODEL_SERVER:
#     client = openai.Client(
#                 base_url=f"{XVERIFY_MATH_MODEL_SERVER}/v1", 
#                 api_key="EMPTY")


import asyncio
class AsyncLLM(object):
    def __init__(self, args):
        import vllm
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        engine_args = vllm.AsyncEngineArgs(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.98,
            dtype="bfloat16",
            disable_log_requests=True,
            seed=args.seed)
        self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        
    async def generate_async(self, prompt, sampling_params, request_id):
        # Send the request to the LLM engine.
        import asyncio
        async with asyncio.Semaphore(1):
            stream = self.async_llm.generate(
                request_id=str(request_id),
                prompt=prompt,
                sampling_params=sampling_params,
            )

            # Consume the stream until the request is finished.
            async for request_output in stream:
                final_output = request_output
            return final_output, request_id
        
    async def batch_generate(self, prompts, sampling_params):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        tasks = []
        for idx, (prompt, sampling_param) in enumerate(zip(prompts, sampling_params)):
            request_id = str(uuid.uuid4()) + f'####idx:{idx}'
            task = self.generate_async(prompt, sampling_param, request_id)
            tasks.append(task)
            
        import random
        random.shuffle(tasks)
            
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        return task_results
        
    def generate(self, prompts, sampling_params):
        task_results = asyncio.run(self.batch_generate(prompts, sampling_params))
        task_results.sort(key=lambda item: int(item[1].split('####idx:')[-1]))
        outputs = []
        for task_result in task_results:
            outputs.append(task_result[0])
        return outputs

import json
from pydantic import BaseModel
from typing import Optional, Any, List, Dict, Tuple
class Data(BaseModel):
    """
    Data class for game/corpus
    @param question: question of the game/corpus
    @param answer: answer of the game/corpus
    @param difficulty: difficulty of the game/corpus, from 1 to 10
    """
    question: str
    answer: Any
    difficulty: int = 1
    metadata: dict = None
    

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)

def evaluation(args, data_name, llm, tokenizer):
    print(f"### being to evaluate {data_name} ###")
    data_list = []
    with open(os.path.join(args.data_dir, data_name)) as frobj:
        for line in tqdm(frobj):
            d = json.loads(line.strip())
            data_list.append(d)
            
    print(data_list[0].keys())
    
    stop_words = ["<|im_end|>", "<|endoftext|>", "</answer>", "</answer>\n"]
    sampling_params = SamplingParams(
                temperature=float(args.temperature),
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens_per_call,
                n=1,
                seed=args.seed,
                stop=stop_words,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            )
    
    print('==sampling_params==', sampling_params)
    
    input_prompts = []
    for d in data_list:
        for q_key in args.input_key.split(','):
            if q_key in d:
                input_prompts.append(d[q_key])
                break
    
    assert len(input_prompts) == len(data_list)
    
    # repeat n times
    prompts = [
        TEMPLATE[args.prompt_type].replace('{input}', prompt) for prompt in input_prompts for _ in range(args.n_sampling)
    ]
    
    prompts_idx = [
        idx for (idx, prompt) in enumerate(input_prompts) for _ in range(args.n_sampling)
    ]
    
    
    if args.use_vllm:
        if args.use_seperate:
            outputs = []
            for prompt in prompts:
                output = llm.generate(
                        [prompt],
                        sampling_params
                    )
                outputs.append(output[0])
        else:
            outputs = llm.generate(
                        prompts,
                        sampling_params
                    )
    elif args.use_vllm_tir:
        if args.use_seperate:
            outputs = []
            for prompt in prompts:
                output = math_tir_generate(llm, sampling_params, None, tokenizer, prompts=[prompt])
                outputs.append(output[0])
            # outputs = math_tir_generate(llm, sampling_params, None, tokenizer, prompts=prompts)
        else:
            outputs = math_tir_generate(llm, sampling_params, None, tokenizer, prompts=prompts)
    
    assert len(outputs) == len(prompts)
    
    for idx in range(len(prompts)):
        d_idx = prompts_idx[idx]
        d = data_list[d_idx]
        if 'pred_response' not in d:
            d['pred_response'] = []
        output = outputs[idx]
        d['pred_response'].append(output.outputs[0].text)
        
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    if args.use_vllm_tir:
        out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_nsample{args.n_sampling}_enviter{ENV_ITER_NUM}_vllm{VLLM_VERSION}"
    else:
        out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_nsample{args.n_sampling}_vllm{VLLM_VERSION}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    
    # Calculate pass@k.
    total, correct = [], []
    for d in data_list:
        d['pred_score'] = []
        d['pred_answer'] = []
        for resp in d['pred_response']:
            d['pred_answer'].append(resp)
            resp_ans = resp.split('</think>')[-1]
            d_label = json.loads(d['label'])
            data = Data(
                question=d['query'],
                answer=d_label['answer'],
                difficulty=d_label['difficulty'],
                metadata=d_label['metadata']
            )
            verifier_name = d_label['data_source'].split('/')[-1]
            # try:
            score = verifier_classes[verifier_name]().verify(data, resp_ans)
            score = float(score)
            # except:
            #     score = 0.0
            d['pred_score'].append(score)
        if args.n_sampling > 1:
            total.append(len(d['pred_score']))
            correct.append(sum(d['pred_score']))
        
    pass_at_k = {}
    if args.n_sampling > 1:
        avg_at_k = {}
        score_at_k = [[] for _ in range(args.n_sampling)]
        for d in data_list:
            assert len(d['pred_score']) == args.n_sampling
            for idx, score in enumerate(d['pred_score']):
                score_at_k[idx].append(score)
        
        avg_score = []
        for sampling_idx in range(args.n_sampling):
            score = 100 / len(data_list) * sum(score_at_k[sampling_idx])
            avg_score.append(score)
        pass_at_k[f'avg@{args.n_sampling}'] = sum(avg_score) / args.n_sampling
        
    return data_list, out_file, pass_at_k
    

def evaluation_main(args):
    
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    enforce_eager = os.getenv('ENFORCE_EAGER', 'FALSE')
    
    print(available_gpus, '==available_gpus==')
    
    if args.use_seperate:
        llm = AsyncLLM(args)
    else:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,gpu_memory_utilization=0.98,
            dtype="bfloat16",
            enforce_eager=True if enforce_eager == 'TRUE' else False,
            seed=args.seed
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    
    avg_score = 0.0
    score_dict = OrderedDict()
    for data_name in args.data_names.split(','):
        score_dict[data_name] = {}
        data_list, out_file, pass_at_k = evaluation(args, data_name, llm, tokenizer)
        if args.n_sampling == 1:
            data_score = sum([d['pred_score'][0] for d in data_list])
            final_score = 100 / len(data_list) * data_score
            score_dict[data_name]['final_score'] = final_score
        
        score_dict[data_name].update(pass_at_k)
        
        with open(out_file, 'w') as fwobj:
            for d in data_list:
                fwobj.write(json.dumps(d, ensure_ascii=False)+'\n')
            
        print(data_name, '===', score_dict[data_name])
        print(data_name, '====', out_file, '==out_file==')
            
    
    data = []
    headers = []
    for name in score_dict:
        item = [name]
        headers = ['dataset']
        for score_key in score_dict[name]:
            item.append(score_dict[name][score_key])
            headers.append(score_key)
        data.append(item)

    table = tabulate(data, headers=headers, tablefmt="pipe")
    
    print(f'### {out_file} evaluation ###')
    print(table)
    
    metric_path = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    with open(metric_path, "w") as f:
        json.dump({
            'value': score_dict,
        }, f, indent=4)
    print(f'### {metric_path} ###')
        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--input_key", default="problem,question", type=str)
    parser.add_argument("--answer_key", default="answer,final_answer", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--pass_at_k", default=1, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max_tokens_per_call", default=16384, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_vllm_tir", action="store_true")
    parser.add_argument("--use_seperate", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    evaluation_main(args)