import numpy as np

def detect_outliers_iqr(data, threshold=1.5, use_lower_bound=False):
    """使用IQR方法检测异常值"""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    if use_lower_bound:
        outlier_idx = np.where((data < lower_bound) | (data > upper_bound))[0]
    else:
        outlier_idx = np.where((data > upper_bound))[0]
    return outlier_idx