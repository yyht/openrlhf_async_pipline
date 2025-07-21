

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from openrlhf.models.utils import masked_mean

class DPGLoss(nn.Module):
    """
    DPG Loss for PPO
    """

    def __init__(self, clip_eps_low: float = 0.2, clip_eps_high: float = 0.2, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        global_response_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = -log_probs * advantages.exp()
        if global_response_length is not None:
            loss = (loss * action_mask).sum(dim=None) / (1e-10+global_response_length)
        else:
            loss = (
                masked_mean(loss, action_mask, dim=None)
                if self.token_level_loss
                else masked_mean(loss, action_mask, dim=-1).mean()
            )
        return loss