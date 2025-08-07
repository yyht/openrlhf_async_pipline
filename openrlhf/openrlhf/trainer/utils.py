

import numpy as np

def repeatness_group(group_samples):
    repeatness_score = [sample.rewards[0]['repeatness'][0] for sample in group_samples]
    mean_score = np.mean(repeatness_score)
    std_score = np.std(repeatness_score, ddof=1)
    # 计算Z分数（取绝对值）
    z_scores = np.abs((data - mean) / std)
    pass