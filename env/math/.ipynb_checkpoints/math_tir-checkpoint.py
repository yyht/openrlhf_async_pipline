

import ray
import os, copy
import uuid

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_0304'))

from env.math.code_exec import run_code
from env.math.extract_code import extract_code
from collections import OrderedDict
import re
code_pattern = re.compile(r"<code>.*?</code>", re.DOTALL)

def math_tir_world_model(llm, sampling_params, prompt_token_ids, tokenizer, prompts=None):
    
    output_token_ids = [[]]*len(prompts)
    action_masks = [[]]*len(prompts)

    id2uuid = OrderedDict()
    uuid2id = OrderedDict()
    uuid2data = OrderedDict()

    for idx, prompt in enumerate(prompts):
        uuid_num = str(uuid.uuid4())
        id2uuid[idx] = uuid_num
        uuid2id[uuid_num] = idx
        uuid2data[uuid_num] = prompt

    is_all_terminated = [False]*len(prompts)
    is_terminated = sum(is_all_terminated) == len(is_all_terminated)
    idx_list = list(range(len(prompts)))

    while not is_terminated:

        outputs = llm.generate(sampling_params=sampling_params, prompts=new_prompts)

        left_idx = []
        left_prompts = []

        for prompt, output, prompt_idx in zip(new_prompts, outputs, idx_list):
            text = output[0].text
            token_ids = output[0].token_ids
            action_mask = [1] * len(token_ids)

            if output[0].stop_reason in ['</code>']:
                # stopped by '</code>'
                code_text = re.findall(code_pattern, text)
                if code_text:
                    code_text = code_text.replace('<code>', '```python')
                    code_text = code_text.replace('</code>', '```')
                    code4exec = extract_code(code_text)
                    if code4exec:
                        try:
                            result = run_code(code4exec)
                        except Exception as e:
                            result = str(e)

                        code_output = f"""```{result}```"""
                        
                        code_output_ids = tokenizer(code_output, add_special_tokens=False)['input_ids']
                        token_ids.extend(code_output_ids)
                        action_mask.extend([0]*len(code_output_ids))
                    else:
                        code_output = ''

                    prompt += code_output
                    left_idx.append(prompt_idx)
                    left_prompts.append(prompt)
            else:
                is_all_terminated[prompt_idx] = True

            output_token_ids[prompt_idx].extend(token_ids)
            action_masks[prompt_idx].extend(action_mask)

        is_terminated = sum(is_all_terminated) == len(is_all_terminated)
        new_prompts = left_prompts
        idx_list = left_idx

        assert len(new_prompts) == len(idx_list)
                
        


    