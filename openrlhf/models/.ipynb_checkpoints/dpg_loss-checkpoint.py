

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
        items_mask: Optional[torch.Tensor] = None,
        length_status: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if DPG_TYPE == 'dpg':
            loss = log_probs * advantages.exp()
        elif DPG_TYPE == 'dpg_baseline':
            importance_ratios = (log_probs.detach().float() - old_log_probs.float())
            importance_ratios = masked_sum(importance_ratios, action_mask, dim=-1)
            advantage = advantages.exp() - importance_ratios.exp()
            loss = log_probs * advantage
        elif DPG_TYPE == 'dpg_f_div':
            importance_ratios = (log_probs.detach().float() - old_log_probs.float())
            importance_ratios = masked_sum(importance_ratios, action_mask, dim=-1)
            # [batch_size, seq] - [batch_size]
            adv = (advantages - importance_ratios).unsqueeze(-1)
            zeros = torch.zeros_like(adv)
            advantage = importance_ratios.exp() * (torch.logsumexp(torch.cat([zeros, adv], dim=-1), dim=-1, keepdim=True) - torch.log(torch.tensor(2.0)))
            loss = log_probs * advantage
        if length_status is not None:
            loss = -masked_sum(loss, action_mask, dim=-1).sum()/(length_status['response_length']+1e-10)
        else:
            loss = -masked_mean(loss, action_mask, dim=-1).mean()
        return loss