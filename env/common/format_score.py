

import re
zero_pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
zero_format_pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)

def format_score_fn_v1(prompt, response, use_format_reward='no'):
    format_matches = re.findall(zero_format_pattern, response)
    matches = re.findall(zero_pattern, response)
    is_valid = False
    final_reward = 0.0
    if len(format_matches) > 0:
        is_valid = False
        final_reward = 0.0
    else:
        if use_format_reward == 'yes':
            final_reward = -1.0
        else:
            final_reward = 0.0
    if len(format_matches) > 0:
        if len(matches) > 0:
            is_valid = True
            final_reward += 0.0
        else:
            if use_format_reward == 'yes':
                final_reward += -0.5
            else:
                final_reward = 0.0
    if is_valid:
        for tag in ['</think>', '<answer>', '</answer>']:
            count = response.count(tag)
            if count != 1:
                is_valid = False
                if use_format_reward == 'yes':
                    final_reward = -2.0
                else:
                    final_reward = 0.0
                break
    if is_valid:
        left_str = ''.join(response.split('</answer>')[1:])
        left_str_len = len(tokenizer(left_str)['input_ids'])
        if left_str_len > 5:
            is_valid = False
            if use_format_reward == 'yes':
                final_reward = -2.0
            else:
                final_reward = 0.0
    return is_valid, final_reward