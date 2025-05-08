

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