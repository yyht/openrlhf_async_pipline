

def uct_selection(experience, prior_weight):
    ff_influence_score = experience.info['ff_influence']
    prior = experience.info['action_ppl']
    rewards = experience.info['answer_rewards']
    self_certainty = experience.info['self_certainty']

    # puct-metric
    q = ((ff_influence_score > 0).float() + rewards) + c_puct * prior