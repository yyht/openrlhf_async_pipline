


def sample_filter_fn(sample, **kwargs):
    label = sample.labels[0]
    repeatness_threshold = label.get('repeatness_threshold', 0.05)
    gold_score = sample.rewards[0]['answer_rewards'][0] > 0.05
    if float(sample.rewards[0]['repeatness'][0]) <= repeatness_threshold and gold_score:
        return 1
    return 0

def overlong_filter_fn(exp, sample, **kwargs):
    args = kwargs['args']
    if sample.rewards[0]['length_rewards'][0] >= args.generate_max_len:
        return True
    return False

def exp_filter_fn(exp, sample, **kwargs):
    label = sample.labels[0]
    repeatness_threshold = label.get('repeatness_threshold', 0.05)
    gold_score = sample.rewards[0]['answer_rewards'][0] > 0.5
    overlong_filter = (overlong_filter_fn(exp, sample, **kwargs) or float(sample.rewards[0]['truncated'][0]) > 0.5) and not gold_score
    positive_repeatness_filter = gold_score and (float(sample.rewards[0]['repeatness'][0]) > 0.2 or float(sample.rewards[0]['repetition_penalty'][0]) > 0.6)
    code_filter = sample.rewards[0]['code_count'][0] != sample.rewards[0]['output_count'][0] or sample.rewards[0]['code_count'][0] != sample.rewards[0]['partial_code_count'][0]
    void_filter = (float(sample.rewards[0]['format_answer'][0]) > 0.5) and (float(sample.rewards[0]['repeatness'][0]) > 0.2 or float(sample.rewards[0]['repetition_penalty'][0]) > 0.6)
    shorcut_filter = gold_score and code_filter
    # code_valid_format_invalid_format_filter = (float(sample.rewards[0]['format_answer'][0]) > 0.5) and not code_filter and sample.rewards[0]['code_exection_all_failed'][0] < 1 and (float(sample.rewards[0]['repeatness'][0]) < repeatness_threshold or float(sample.rewards[0]['repetition_penalty'][0]) < 0.5)
    if exp is not None:
        large_ppl = exp.info['base_ppl'][0] > 5
    else:
        large_ppl = False
    code_valid_filter = (float(sample.rewards[0]['format_answer'][0]) < 0.5) and code_filter and sample.rewards[0]['code_exection_all_failed'][0] < 1
    truncated_filter = float(sample.rewards[0]['other'][0]) > 0.5
    # return truncated_filter or positive_repeatness_filter or void_filter or shorcut_filter or large_ppl or code_valid_filter
    # code_valid_filter = (float(sample.rewards[0]['format_answer'][0]) > 0.5) and code_filter and sample.rewards[0]['code_exection_all_failed'][0] < 1 and float(sample.rewards[0]['repeatness'][0]) < repeatness_threshold and float(sample.rewards[0]['repetition_penalty'][0]) < 0.5
    return overlong_filter or truncated_filter or shorcut_filter or code_valid_filter or large_ppl or void_filter

def reward_fail_fn(sample, **kwargs):
    if sample.rewards[0]['rule_eval_fails'][0] > 0.5 and sample.rewards[0]['model_eval_fails'][0] > 0.5:
        return 1
    return 0