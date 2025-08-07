

import torch

def clip_by_abs(input_tensor: torch.Tensor, clip_low: float = 0.0, clip_high: float = 1.0) -> torch.Tensor:
    clipped_tensor = torch.clamp(input_tensor, min=-1-clip_low, max=1+clip_high)
    return clipped_tensor