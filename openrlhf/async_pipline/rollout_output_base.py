

from typing import Generic, TypeVar, Union, NamedTuple
from typing import Optional, Any, List, Dict, Tuple

class Output(NamedTuple):
    token_ids: list[int]
    action_mask: list[int]
    text: str
    stop_reason: str
    finish_reason: str
    env_exec_times: int
    reward_info: Optional[dict] = {}

class GenerateOutput(NamedTuple):
    outputs: list[Output]
    prompt_token_ids: list[int]
    request_id: str
    label: Optional[dict] = {}
    prompt: Optional[str] = ''
    request_rank: Optional[int] = 0

# 转换函数
def generate_output_to_dict(generate_output: GenerateOutput) -> dict:
    # 转换每个Output对象为字典
    generate_dict = generate_output._asdict()
    outputs_dict = [output._asdict() for output in generate_output.outputs]
    generate_output_dict = {}
    for key in generate_dict:
        if key in ['outputs']:
            generate_output_dict[key] = outputs_dict
        else:
            generate_output_dict[key] = generate_dict[key]
    # 构建最终字典
    return generate_output_dict

def generate_output_dict2GenerateOutput(generate_output_dict: dict) -> GenerateOutput:
    # 转换每个输出字典为Output对象
    outputs = []
    for output_dict in generate_output_dict.get('outputs', []):
        output = Output(
            token_ids=output_dict.get('token_ids', []),
            action_mask=output_dict.get('action_mask', []),
            text=output_dict.get('text', ''),
            stop_reason=output_dict.get('stop_reason', ''),
            finish_reason=output_dict.get('finish_reason', ''),
            env_exec_times=output_dict.get('env_exec_times', 0),
            reward_info=output_dict.get('reward_info', {})
        )
        outputs.append(output)
    
    # 构建并返回GenerateOutput对象
    return GenerateOutput(
        outputs=outputs,
        prompt_token_ids=generate_output_dict.get('prompt_token_ids', []),
        request_id=generate_output_dict.get('request_id', ''),
        label=generate_output_dict.get('label', {}),
        prompt=generate_output_dict.get('prompt', ''),
        request_rank=generate_output_dict.get('request_rank', 0)
    )

