
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean

class RegGRPOLoss(nn.Module):
    """
    Policy Loss for PPO
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
        pass