

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean, masked_sum

import os
DPG_TYPE = os.getenv('DPG_TYPE', 'dpg')

class DPGLoss(nn.Module):
    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        env_mask: Optional[torch.Tensor] = None,
        length_status: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if DPG_TYPE == 'dpg':
            loss = log_probs * advantages.exp()
        elif DPG_TYPE == 'dpg_baseline':
            importance_ratios = (log_probs.detach() - old_log_probs)
            importance_ratios = masked_sum(importance_ratios, action_mask, env_mask, dim=-1)
            advantage = advantages.exp() - importance_ratios.exp()
            loss = log_probs * advantage
        elif DPG_TYPE == 'dpg_f_div':
            importance_ratios = (log_probs.detach() - old_log_probs)
            importance_ratios = masked_sum(importance_ratios, action_mask, env_mask, dim=-1)

            # u = \frac{\pi_{theta}(x)}{P(x)}
            # [batch_size, 1]
            
            log_u = (masked_sum(log_probs.detach(), action_mask, env_mask, dim=-1) - advantages).unsqueeze(dim=-1)
            # [batch_size, 2]
            log_u_zero = torch.concat([torch.zeros_like(log_u), log_u], dim=-1)
            log_u_one = torch.logsumexp(log_u_zero, dim=-1)
            adv = log_u - log_u_one + torch.log(torch.tensor(2.0))

            advantage = importance_ratios.exp() * adv
            loss = log_probs * advantage
        if length_status is not None:
            loss = -masked_sum(loss, action_mask, env_mask, dim=-1) / (length_status['response_length']+1e-10)
        else:
            loss = -masked_mean(loss, action_mask, env_mask, dim=-1).mean()
        return loss