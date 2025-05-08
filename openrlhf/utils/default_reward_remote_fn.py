
def default_remote_fn(data):
    default = {
        'rewards': [0.0],
        'length_rewards': [0.0],
        'answer_rewards': [0.0],
        'reflection': [0.0],
        'new_idea': [0.0],
        'code_rewards': [0.0],
        'code_correct': [0.0],
        'rule_eval_fails': [1.0],
        'model_eval_fails': [1.0],
        'more_boxed': [0.0],
        'no_boxed': [0.0],
        "format_answer": [0.0],
        "finish_reason": [0.0],
        'truncated': [0.0],
        'other': [0.0],
        "repeatness": [0.0],
        "code_count": [0.0],
        'model_rewards': [0.0],
        'rule_rewards': [0.0]
    }
    default_tmp = {}
    for idx, _ in enumerate(data['query']):
        for key in default:
            if key not in default_tmp:
                default_tmp[key] = [default[key][0]]
            else:
                default_tmp[key].append(default[key][0])
    output_dict = {
        'rewards': default_tmp
    }
    return output_dict
