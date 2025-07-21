

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))
from env.math.math_tir import math_tir_generate
from env.math.math_tir_process_single_request import math_tir_generate_async
# from env.math.math_tir_process_single_request_v1 import math_tir_generate_async as math_tir_generate_async_v1

ENV_GENERATE_CONFIG = {
    'math_tir_generate': math_tir_generate,
    'math_tir_async': math_tir_generate_async,
    # 'math_tir_async': math_tir_generate_async_v1,
}