
from collections import Counter
from openrlhf.trainer.ppo_utils.exp_balancing import select_optimal_tasks
from openrlhf.utils.logging_utils import init_logger
import numpy as np

logger = init_logger(__name__)
def sample_statistics(samples):
    task_statistics = {}
    for sample in samples:
        task = sample.labels[0]['task'].lower()
        if task not in task_statistics:
            task_statistics[task] = {
                'positive': {
                    'count': 0
                },
                'negative': {
                    'count': 0
                },
            }
        if sample.rewards[0]['answer_rewards'][0] > 0.05:
            task_statistics[task]['positive']['count'] += 1
            for key in sample.rewards[0]:
                if sample.rewards[0][key]:
                    if isinstance(sample.rewards[0][key][0], list):
                        continue
                    if key not in task_statistics[task]['positive']:
                        task_statistics[task]['positive'][key] = []
                    task_statistics[task]['positive'][key].append(sample.rewards[0][key][0])
        else:
            task_statistics[task]['negative']['count'] += 1
            for key in sample.rewards[0]:
                if sample.rewards[0][key]:
                    if isinstance(sample.rewards[0][key][0], list):
                        continue
                    if key not in task_statistics[task]['negative']:
                        task_statistics[task]['negative'][key] = []
                    task_statistics[task]['negative'][key].append(sample.rewards[0][key][0])
    
    for task in task_statistics:
        for key in task_statistics[task]['positive']:
            if isinstance(task_statistics[task]['positive'][key], list):
                task_statistics[task]['positive'][key] = sum(task_statistics[task]['positive'][key]) / (len(task_statistics[task]['positive'][key])+1e-10)
        for key in task_statistics[task]['negative']:
            if isinstance(task_statistics[task]['negative'][key], list):
                task_statistics[task]['negative'][key] = sum(task_statistics[task]['negative'][key]) / (len(task_statistics[task]['negative'][key])+1e-10)
    return task_statistics

def sample_strategy(samples, args):
    experience_num = args.rollout_batch_size * args.n_samples_per_prompt
    target_group_size = experience_num // args.n_samples_per_prompt
    group_samples = {}
    for sample in samples:
        request_id = sample.request_ids[0].split('####idx:')[0]
        if request_id not in group_samples:
            group_samples[request_id] = []
        group_samples[request_id].append(sample)
    group_keys = list(group_samples.keys())
    logger.info({
        'INFO': '##SAMPLE-STRATEGY##',
        'VALUE': len(group_samples),
        'GROUP_SIZE': len(samples) // args.n_samples_per_prompt
    })
    if args.sample_strategy == 'random':
        import random
        random.shuffle(group_keys)
        candidate_samples = sum([group_samples[key] for key in group_keys[:target_group_size]], [])
        left_samples = sum([group_samples[key] for key in group_keys[target_group_size:]], [])
        return candidate_samples, left_samples
    elif args.sample_strategy == 'acc_balancing':
        group_acc = {}
        for key in group_samples:
            if key not in group_acc:
                group_acc[key] = []
            group_acc[key] = [sample.rewards[0]['answer_rewards'][0] for sample in group_samples[key]]
        
        best_selection, _ = select_optimal_tasks(group_acc, target_group_size)
        candidate_samples = sum([group_samples[key] for key in best_selection], [])
        left_samples = sum([group_samples[key] for key in group_keys if key not in best_selection], [])
        return candidate_samples, left_samples
    elif args.sample_strategy == 'length_balancing':
        group_length = []
        for key in group_samples:
            positive_length = [sample.response_length.item() for sample in group_samples[key] if sample.rewards[0]['answer_rewards'][0] > 0.5]
            negative_length = [sample.response_length.item() for sample in group_samples[key] if sample.rewards[0]['answer_rewards'][0] < 0.5]
            positive_length_mean = np.mean(positive_length)
            negative_length_mean = np.mean(negative_length)
            length_gap = abs(positive_length_mean-negative_length_mean)
            group_length.append((key, length_gap))
        group_length = sorted(group_length, key=lambda x: x[1], reverse=False)
        candidate_samples = sum([group_samples[key] for key in group_keys[:target_group_size]], [])
        left_samples = sum([group_samples[key] for key in group_keys[target_group_size:]], [])
        return candidate_samples, left_samples
    else:
        import random
        random.shuffle(group_keys)
        candidate_samples = sum([group_samples[key] for key in group_keys[:target_group_size]], [])
        left_samples = sum([group_samples[key] for key in group_keys[target_group_size:]], [])
        return candidate_samples, left_samples

def length_filter(samples, args):
    group_samples = {}
    for sample in samples:
        request_id = sample.request_ids[0].split('####idx:')[0]
        if request_id not in group_samples:
            group_samples[request_id] = []
        group_samples[request_id].append(sample)
    group_keys = list(group_samples.keys())
    logger.info({
        'INFO': '##SAMPLE-STRATEGY##',
        'VALUE': len(group_samples),
        'GROUP_SIZE': len(samples) // args.n_samples_per_prompt
    })
    group_length = []
    for key in group_samples:
        positive_length = [sample.response_length.item() for sample in group_samples[key] if sample.rewards[0]['answer_rewards'][0] > 0.5]
        negative_length = [sample.response_length.item() for sample in group_samples[key] if sample.rewards[0]['answer_rewards'][0] < 0.5]
        positive_length_mean = np.mean(positive_length)
        negative_length_mean = np.mean(negative_length)
        length_gap = abs(positive_length_mean-negative_length_mean)
        group_length.append((key, length_gap))
    group_length = sorted(group_length, key=lambda x: x[1], reverse=False)
    group_filtered = sum([group_samples[x[0]] for x in group_length if x[1] < 1000], [])
    logger.info({
        'INFO': '##SAMPLE-LENGTH-FILTER##',
        'AFTER_FILTER_VALUE': len(group_filtered),
        'BEFORE_FILTER_VALUE': len(samples)
    })
    return group_filtered

def length_balance(samples):
    group_samples = {}
    for sample in samples:
        request_id = sample.request_ids[0].split('####idx:')[0]
        if request_id not in group_samples:
            group_samples[request_id] = []
        group_samples[request_id].append(sample)
    
    length_acc = {}
    for length in range(0, 32000, 1000):
        length_acc[length] = []

    # 将样本按长度分组
    for request_id in group_samples:
        for sample in group_samples[request_id]:
            sample_length = sample.action_mask.sum().item()
            # 确定样本所属的长度区间
            for length_threshold in sorted(length_acc.keys()):
                if sample_length < length_threshold:
                    length_acc[length_threshold].append({
                        'request_id': request_id,
                        'rewards': sample.rewards[0]['answer_rewards'][0]
                    })
                    break  # 找到合适区间后跳出循环
    
    last_acc_balance_length = 0
    valid_length = set()
    for length_threshold in sorted(length_acc.keys(), reverse=True):
        info = length_acc[length_threshold]
        if info:
            info_positive = sum([item['rewards'] for item in info])
            ratio = info_positive / (len(info)+1e-10)
            if ratio >= 0.05:
                valid_length.add(length_threshold)
    valid_request_id = set()
    for length_threshold in valid_length:
        for info in length_acc[length_threshold]:
            valid_request_id.add(info['request_id'])
    
    output_samples = []
    for request_id in valid_request_id:
        output_samples.extend(group_samples[request_id])

    logger.info({
        'INFO': '##GROUP-LENGTH-ACC-BALANCING##',
        'VALUE': f"BEFORE: {len(samples)};AFTER: {len(output_samples)}"
    })
    
    return output_samples
        

            
    

