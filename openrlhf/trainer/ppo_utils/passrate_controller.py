import numpy as np


class AdaptiveController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_acc_low):
        self.value = init_acc_low

    def update(self, current, n_steps):
        a = self.value - (n_steps // 10) * 0.1
        return max(0.1, a)    


class FixedController:
    """Fixed KL controller."""

    def __init__(self, init_acc_low):
        self.value = init_acc_low

    def update(self, current, n_steps):
        return self.value
