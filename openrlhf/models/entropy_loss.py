
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean, masked_sum

def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

class EntropyLoss(nn.Module):
    """
    Entropy Loss for PPO
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        action_mask: Optional[torch.Tensor] = None,
        env_mask: Optional[torch.Tensor] = None,
        length_status: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [batch_size, response_length, vocab_size]
        action_logits = logits[:, :-1, :]
        action_logits = action_logits[:, -num_actions:, :]
        # [batch_size, response_length]
        entropy = entropy_from_logits(action_logits)
        if length_status is not None:
            entropy_loss = masked_sum(entropy, action_mask, env_mask, dim=-1).sum() / (length_status['response_length']+1e-10)
        else:
            entropy_loss = masked_mean(entropy, action_mask, env_mask, dim=-1).mean()
        return entropy_loss