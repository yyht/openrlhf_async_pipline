

from openrlhf.models.actor import Actor
from openrlhf.trainer.ppo_utils.experience_maker import Samples
from openrlhf.trainer.ppo_utils.replay_buffer import NaiveReplayBuffer

pretrained_models = '/newcpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/'
ckpt = '/newcpfs/user/chenhao/outputs/qwen25_7B_reinforce_baseline_zero_tir_lr1e-6_warmup0.0_kl0.0_zero_0526_latest_agent_orz_iternum2_text_v0_fix_eos_filter_ori_ds0169_082_global_token_level_loss_clean/global_step250_hf/'

initial_model = Actor(pretrained_models,  use_flash_attention_2=True, packing_samples=True).to('cuda')
actor_model = Actor(ckpt,  use_flash_attention_2=True, packing_samples=True).to('cuda')

sample_path = '/newcpfs/user/chenhao/outputs/qwen25_7B_reinforce_baseline_zero_tir_lr1e-6_warmup0.0_kl0.0_zero_0526_latest_agent_orz_iternum2_text_v0_fix_eos_filter_ori_ds0169_082_global_token_level_loss_clean/rollout_sample_eposide_632.jsonl'

import json
df = []
with open(sample_path) as frobj:
    for line in frobj:
        df.append(json.loads(line.strip()))

samples = []
for d in df:
    for key in d:
        d[key] = torch.tensor(d[key])
    samples.append(Samples(**d))





