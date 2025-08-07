

TEMPLATE = {
    'orz_tir': 
    """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. In your reasoning-process, You can use python-code to solve your problem. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:{input}\nAssistant: <think>""",
    "orz_tir_xinji": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You can use Python code during the solution process, and the code will be executed immediately and the result will be returned. You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:{input}\nAssistant: <think>""",
    "orz_xinji": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:{input}\nAssistant: <think>""",
    "orz_ch": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:{input}\nAssistant: <think>""",
    "qwen25-math-cot-tora": """<|im_start|>system\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n""",
    "deepseek_r1_distill": """<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You should think step-by-step.<｜User｜>{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜>"""
}

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
USE_ID = os.getenv('USE_ID', 'NONE')

sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))
# from env.math.math_tir import math_tir_generate
from env.math.math_tir_process_single_request import math_tir_generate_async
from openrlhf.async_pipline.process_request import GenerateRequest, default_generate, process_batch_requests
from passk_eval import estimate_pass_at_k
from tabulate import tabulate
import uuid

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
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.semaphore = asyncio.Semaphore(512)  # 实例级共享
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.args = args
        self.batch_size = 4
        
    def shutdown(self):
        self.llm.shutdown()  # 释放 GPU 内存
        
    async def generate_async_server(self, request: GenerateRequest, sampling_params, request_id):
        # Send the request to the LLM engine.
        from vllm.inputs import TokensPrompt
        async with self.semaphore:  # 使用共享信号量
        # async with asyncio.Semaphore(MAX_CONCURRENT):  # 实例级共享
            # stream = self.llm.generate(
            #     request_id=str(request_id),
            #     prompt=request.prompts[0],
            #     sampling_params=sampling_params,
            # )
            
            if USE_ID == 'USE_ID':
                stream = self.llm.generate(
                    request_id=str(request_id),
                    prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
                    # prompt=request.prompts[0],
                    sampling_params=sampling_params,
                )
                
            else:
                stream = self.llm.generate(
                    request_id=str(request_id),
                    # prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
                    prompt=request.prompts[0],
                    sampling_params=sampling_params,
                )

            # Consume the stream until the request is finished.
            # 移入循环内部确保作用域隔离
            final_output = None
            async for request_output in stream:
                final_output = request_output
            if final_output is None:
                raise RuntimeError(f"Empty stream for request_id: {request_id}")
            
            assert final_output.request_id == request_id
            output = [{
                'outputs':[
                    {
                        "text": final_output.outputs[0].text,
                        "token_ids": final_output.outputs[0].token_ids,
                        "stop_reason": final_output.outputs[0].stop_reason,
                        "finish_reason": final_output.outputs[0].finish_reason,
                        "log_probs": final_output.outputs[0].logprobs
                    }
                ],
                "prompt_token_ids": final_output.prompt_token_ids,
                "request_id": final_output.request_id
            }]
            return output

    async def async_llm_generate(self, request: GenerateRequest):
        # 实际生成逻辑
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=1.0,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            include_stop_str_in_output=request.include_stop_str_in_output,
            stop=request.stop,
            skip_special_tokens=False,
            logprobs=None
        )

        # request_id = str(uuid.uuid4())+request.uuids
        request_id = f"{time.time_ns()}-{uuid.uuid4()}"
        response = await self.generate_async_server(request, sampling_params, request_id)
        return response
    
    def build_requests(self, prompts, uuids, sampling_params, infer_type='math_tir_async'):
        request_list = []
        for idx, (prompt, uuid_str) in enumerate(zip(prompts, uuids)):
            request = GenerateRequest(
                prompts=[prompt],
                prompt_token_ids=self.tokenizer(prompt)['input_ids'],
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                stop=sampling_params.stop,
                uuids=uuid_str+f'####idx:{idx}',
                env_func=infer_type,
                label=json.dumps({}, ensure_ascii=False),
                request_rank=0,
                max_length=sampling_params.max_tokens+1024,
                enable_vllm_is_correction=False
            )
            request_list.append(request)
        print(len(request_list), '==request_list==')
        return request_list
    
    def _create_batches(self, data_list):
        """将数据分成 batch，返回 [(start_idx, batch), ...]"""
        batches = []
        if isinstance(data_list, list):
            for i in range(0, len(data_list), self.batch_size):
                batch = data_list[i:i + self.batch_size]
                batches.append((i, batch))
            if i + self.batch_size < len(data_list) - 1:
                batches.append((i+1, data_list[i + self.batch_size:]))
        elif isinstance(data_list, dict):
            for env_func in data_list:
                for i in range(0, len(data_list[env_func]), self.batch_size):
                    batch = data_list[env_func][i:i + self.batch_size]
                    batches.append((i, batch))
                if i + self.batch_size < len(data_list[env_func]) - 1:
                    batches.append((i+1, data_list[env_func][i + self.batch_size:]))
        else:
            raise ValueError("data_list must be a list or dict")
        return batches
        
    async def batch_generate(self, prompts, uuids, sampling_params):
        request_list = self.build_requests(prompts, uuids, sampling_params)
        batches = self._create_batches(request_list)
        response_tasks = []
        for start_idx, batch in batches:
            env_func = batch[0].env_func
            response_tasks.append(process_batch_requests(self.async_llm_generate, start_idx, batch, env_func=env_func, tokenizer=self.tokenizer, use_reward=False))

        results_raw = await asyncio.gather(*response_tasks)
            
        flat_results = []
        for result_raw in results_raw:
            successful_results, failed_results = result_raw
            for item in successful_results:
                flat_results.append(item)
        responses = [result[1][1] for result in flat_results]
        responses.sort(key=lambda x: int(x.request_id.split('####idx:')[-1]))
        return responses
        
    def generate(self, prompts, uuids, sampling_params):
        responses = asyncio.run(self.batch_generate(prompts, uuids, sampling_params))
        return responses
    
    

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
    with open(os.path.join(args.data_dir, data_name, 'test.jsonl')) as frobj:
        for line in tqdm(frobj):
            d = json.loads(line.strip())
            for ans_key in args.answer_key.split(','):
                if ans_key in d:
                    d['answer'] = d[ans_key]
                    break
            assert 'answer' in d
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
    
    uuids = []
    for (idx, prompt) in enumerate(input_prompts):
        for _ in range(args.n_sampling):
            uuid_str = str(uuid.uuid4())
            uuids.append(uuid_str)
    
    if args.use_vllm:
        outputs = llm.generate(
                    prompts,
                    sampling_params
                )
    # elif args.use_vllm_tir:
    #     outputs = math_tir_generate(llm, sampling_params, None, tokenizer, prompts=prompts)
        
    if args.use_vllm_tir and args.use_seperate:
        outputs = llm.generate(prompts, uuids, sampling_params)
    
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
            pred_ans = extract_answer(resp)
            if pred_ans:
                d['pred_answer'].append(pred_ans)
            else:
                d['pred_answer'].append('')
            score = answer_grader(str(d['answer']), pred_ans)
            d['pred_score'].append(score)
        
        if args.n_sampling > 1:
            # valid_answer = [pred_ans for pred_ans in d['pred_answer'] if pred_ans]
            # d['pred_maj_answer'] = max(set(valid_answer),
            #                            key=valid_answer.count)
            # d['pred_max_score'] = max(d['pred_score'])
            # d['pred_maj_score'] = answer_grader(str(d['answer']), d['pred_maj_answer'])
        
            total.append(len(d['pred_score']))
            correct.append(sum(d['pred_score']))
            
    if args.n_sampling > 1:
            
        total = np.array(total)
        correct = np.array(correct)

        ks = [int(args.pass_at_k)]
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                     for k in ks if (total >= k).all()}
        
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
                
    else:
        pass_at_k = {}
        
    return data_list, out_file, pass_at_k

def extract_answer(pred_str):
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
        return pred
    else:
        return None
    
@timeout(10, use_signals=False)
def my_verify(gold, pred):
    return float(verify(gold, pred))

def answer_grader(gold_ans, pred_ans):
    
    if pred_ans is None:
        return 0
    
    gold_parsed = parse('\\boxed{'+gold_ans+'}', 
        extraction_mode="first_match", 
        extraction_config=[LatexExtractionConfig()])
    
    pred_parsed = parse(
                "\\boxed{"+pred_ans+"}",
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
    
    if len(gold_parsed) != 0 and len(pred_parsed) != 0:
        try:
            score = my_verify(gold_parsed, 
                                 pred_parsed)
        except Exception as e:
            score = 0
    else:
        score = 0
        
    return score
    

def evaluation_main(args):
    
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    enforce_eager = os.getenv('ENFORCE_EAGER', 'FALSE')
    
    print(available_gpus, '==available_gpus==')
    
    if args.use_seperate:
        print('==using async-llm==')
        llm = None
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
        if args.use_seperate:
            if llm is not None:
                llm.shutdown()
                del llm
            llm = AsyncLLM(args)
        data_list, out_file, pass_at_k = evaluation(args, data_name, llm, tokenizer)
        if args.n_sampling == 1:
            data_score = sum([d['pred_score'][0] for d in data_list])
            final_score = 100 / len(data_list) * data_score
            score_dict[data_name]['final_score'] = final_score
#         else:
#             data_max_score = sum([d['pred_max_score'] for d in data_list])
#             final_max_score = 100 / len(data_list) * data_max_score
#             score_dict[data_name]['final_max_score'] = final_max_score
            
#             data_maj_score = sum([d['pred_maj_score'] for d in data_list])
#             final_maj_score = 100 / len(data_list) * data_maj_score
#             score_dict[data_name]['final_maj_score'] = final_maj_score
        
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