

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

from env.math.math_reward import math_score
from env.math.math_reward_tir import math_score as  math_score_tir
from env.logic.kk_reward import kk_score
from env.logic.zebralogic_reward import zebralogic_score
from env.synlogic.synlogic_reward import synlogic_score

REWARD_CONFIG = {
    'math': math_score,
    'math_tir': math_score_tir,
    'kk': kk_score,
    'zebralogic': zebralogic_score,
    'synlogic': synlogic_score
}