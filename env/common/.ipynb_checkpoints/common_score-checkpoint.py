


import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', 'YOUR_PATH'))

from env.common.utils import make_async
from env.common.zero_template_format_score import zero_template_format_score_fn, zero_tir_template_format_score_fn, think_answer_pattern_score
from env.common.code_pattern import has_code
from env.common.repeatness import repeatness
from env.common.key_pattern import pattern_count
from env.common.repetition_penalty_reward import repetition_penalty_reward
from env.common.length_penalty import soft_overlong_punishment


from concurrent.futures.thread import ThreadPoolExecutor
common_executor = ThreadPoolExecutor(max_workers=1)

zero_template_format_score_fn_async = make_async(
    zero_template_format_score_fn, executor=common_executor)

zero_tir_template_format_score_fn_async = make_async(
    zero_tir_template_format_score_fn, executor=common_executor)

has_code_fn_async = make_async(
    has_code, executor=common_executor)

repeatness_score_fn_async = make_async(
    repeatness, executor=common_executor)

pattern_count_fn_async = make_async(
    pattern_count, executor=common_executor)

repetition_penalty_reward_async = make_async(
    repetition_penalty_reward, executor=common_executor)

think_answer_pattern_score_async = make_async(
    think_answer_pattern_score, executor=common_executor)

soft_overlong_punishment_async = make_async(
    soft_overlong_punishment, executor=common_executor)

COMMON_METRIC = {
    'ZERO_TIR': zero_tir_template_format_score_fn_async,
    'ZERO_V2': zero_template_format_score_fn_async,
    'code_rewards': has_code_fn_async,
    'repeatness': repeatness_score_fn_async,
    'pattern_count': pattern_count_fn_async,
    'repetition_penalty': repetition_penalty_reward_async,
    'think_answer_pattern': think_answer_pattern_score_async,
    'soft_overlong_punishment': soft_overlong_punishment_async
}



