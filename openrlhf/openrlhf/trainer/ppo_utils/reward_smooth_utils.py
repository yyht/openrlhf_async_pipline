

import numpy as np

def reward_smooth(samples, smooth_method='none'):
    if smooth_method == 'none':
        return samples
    
    if smooth_method == 'gaussian':
        pass
