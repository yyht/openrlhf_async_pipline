

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

from env.math.math_filter_fn_utils import sample_filter_fn as math_sample_filter_fn
from env.math.math_filter_fn_utils import exp_filter_fn as math_exp_filter_fn
from env.math.math_filter_fn_utils import reward_fail_fn as math_reward_fail_fn

from env.math.math_tir_filter_fn_utils import sample_filter_fn as math_tir_sample_filter_fn
from env.math.math_tir_filter_fn_utils import exp_filter_fn as math_tir_exp_filter_fn
from env.math.math_tir_filter_fn_utils import reward_fail_fn as math_tir_reward_fail_fn

from env.synlogic.filter_fn_utils import sample_filter_fn as synlogic_sample_filter_fn
from env.synlogic.filter_fn_utils import exp_filter_fn as synlogic_exp_filter_fn
from env.synlogic.filter_fn_utils import reward_fail_fn as synlogic_reward_fail_fn

FILTER_FN_CONFIG = {
    'math_sample_filter_fn': math_sample_filter_fn,
    'math_exp_filter_fn': math_exp_filter_fn,
    'math_reward_fail_fn': math_reward_fail_fn,
    'math_tir_sample_filter_fn': math_tir_sample_filter_fn,
    'math_tir_exp_filter_fn': math_tir_exp_filter_fn,
    'math_tir_reward_fail_fn': math_tir_reward_fail_fn,
    'synlogic_sample_filter_fn': synlogic_sample_filter_fn,
    'synlogic_exp_filter_fn': synlogic_exp_filter_fn,
    'synlogic_reward_fail_fn': synlogic_reward_fail_fn,
}