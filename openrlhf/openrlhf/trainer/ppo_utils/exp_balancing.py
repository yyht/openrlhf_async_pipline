
def balanced_subset(lst, k):
    n = len(lst)
    if k > n:
        k = n  # 如果k大于列表长度，则取整个列表
    if k == 0:
        return [], []  # 返回空列表和原列表
    
    # 统计0和1的数量
    count0 = lst.count(0)
    count1 = n - count0
    
    # 计算0的最小和最大可能数量
    low_bound = max(0, k - count1)  # 至少需要这么多0
    high_bound = min(count0, k)     # 至多能取这么多0
    
    # 候选0的数量：k//2, (k+1)//2, 以及边界值
    candidates = {k // 2, (k + 1) // 2, low_bound, high_bound}
    # 过滤出在可行范围内的候选值
    valid_candidates = [x for x in candidates if low_bound <= x <= high_bound]
    
    # 选择最优的0的数量（使|2x - k|最小，相同最小值时选较大的x）
    best_x = None
    best_value = float('inf')
    for x in valid_candidates:
        value = abs(2 * x - k)  # 衡量均衡度的目标函数
        if value < best_value:
            best_value = value
            best_x = x
        elif value == best_value:
            if x > best_x:  # 相同均衡度时，选0的数量更多
                best_x = x
    
    x0 = best_x
    y0 = k - x0  # 1的数量
    
    # 收集0和1的索引
    zeros_indices = []
    ones_indices = []
    for idx, num in enumerate(lst):
        if num == 0:
            zeros_indices.append(idx)
        else:
            ones_indices.append(idx)
    
    if len(ones_indices) < 1:
        return [], []
    
    # 选取前x0个0和前y0个1的索引
    selected_zeros = zeros_indices[:x0]
    selected_ones = ones_indices[:y0]
    
    # 合并索引并排序以保持原顺序
    selected_indices = sorted(selected_zeros + selected_ones)
    
    # 计算剩余元素
    remaining_indices = [i for i in range(n) if i not in selected_indices]
    
    return selected_indices, remaining_indices

import random
from collections import defaultdict

def select_optimal_tasks(tasks, task_num, max_iterations=1000, use_heuristic=True):
    """
    使用贪心算法选择指定数量的任务，使它们的0和1元素尽可能相等
    
    参数:
    tasks (dict): 任务字典，格式为 {"task1": [0,1,0,1], "task2": [1,1,0,0], ...}
    task_num (int): 需要选择的任务数量
    max_iterations (int): 最大迭代次数
    use_heuristic (bool): 是否使用启发式初始化
    
    返回:
    list: 最优任务组合
    int: 最优组合的0和1数量的最小绝对差
    """
    # 检查输入是否有效
    if not tasks or task_num <= 0 or task_num > len(tasks):
        return [], float('inf')
    
    task_names = list(tasks.keys())
    num_columns = len(next(iter(tasks.values())))
    
    if use_heuristic:
        # 启发式初始化：选择差异最大的任务
        task_diffs = []
        for name in task_names:
            ones_count = sum(tasks[name])
            zeros_count = num_columns - ones_count
            task_diffs.append((abs(ones_count - zeros_count), name))
        
        # 按差异排序，选择差异最大的任务作为初始集
        task_diffs.sort(reverse=True)
        initial_tasks = [task_diffs[i][1] for i in range(min(task_num, len(task_diffs)))]
        current_selection = set(initial_tasks)
    else:
        # 随机初始化
        current_selection = set(random.sample(task_names, task_num))
    
    # 计算初始列和
    col_sums = [0] * num_columns
    for task_name in current_selection:
        for i, val in enumerate(tasks[task_name]):
            col_sums[i] += val
    
    best_selection = current_selection.copy()
    min_diff = calculate_diff(col_sums, task_num, num_columns)
    no_improvement = 0
    
    # 模拟退火式的贪心算法
    for iteration in range(max_iterations):
        if no_improvement > max_iterations // 10:
            break  # 提前终止条件
        
        # 随机选择一个任务移除
        remove_candidate = random.sample(current_selection, 1)[0]
        
        # 计算移除该任务后的列和
        new_col_sums = list(col_sums)
        for i, val in enumerate(tasks[remove_candidate]):
            new_col_sums[i] -= val
        
        # 评估所有可能的替换任务
        best_replacement = None
        best_replacement_diff = float('inf')
        
        for add_candidate in task_names:
            if add_candidate in current_selection:
                continue
                
            # 计算添加该任务后的列和
            candidate_col_sums = list(new_col_sums)
            for i, val in enumerate(tasks[add_candidate]):
                candidate_col_sums[i] += val
            
            # 计算差异
            candidate_diff = calculate_diff(candidate_col_sums, task_num, num_columns)
            
            if candidate_diff < best_replacement_diff:
                best_replacement_diff = candidate_diff
                best_replacement = add_candidate
        
        # 如果找到更好的替换方案，则更新
        if best_replacement_diff < min_diff:
            # 执行替换
            current_selection.remove(remove_candidate)
            current_selection.add(best_replacement)
            col_sums = [new_col_sums[i] + tasks[best_replacement][i] for i in range(num_columns)]
            min_diff = best_replacement_diff
            best_selection = current_selection.copy()
            no_improvement = 0
        else:
            no_improvement += 1
    
    return list(best_selection), min_diff

def calculate_diff(col_sums, task_num, num_columns):
    """计算0和1的数量差"""
    total_ones = sum(col_sums)
    total_zeros = num_columns * task_num - total_ones
    return abs(total_ones - total_zeros)