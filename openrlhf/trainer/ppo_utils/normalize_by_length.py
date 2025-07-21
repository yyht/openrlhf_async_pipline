
import torch
def normalize_experience_by_length(experiences, args):
    group_experiences = {}
    for experience in experiences:
        request_id = experience.request_ids[0].split('####idx:')[0]
        if request_id not in group_experiences:
            group_experiences[request_id] = []
        group_experiences[request_id].append(experience)

    length_acc = {}
    for length in range(0, 32000, 1000):
        length_acc[length] = []
    
    for request_id, experiences in group_experiences.items():
        for experience in experiences:
            sample_length = experience.action_mask.sum().item()
            for length_threshold in sorted(length_acc.keys()):
                if sample_length < length_threshold:
                    length_acc[length].append(experience)
                    break
    
    output_experiences = []
    for length in length_acc:
        if not length_acc[length]:
            continue
        all_advantages = []
        all_action_masks = []
        for exp in length_acc[length]:
            all_advantages.append(exp.advantages.flatten())
            all_action_masks.append(exp.action_mask.flatten())

        advantages_vector = torch.cat(all_advantages, dim=0).float()
        action_masks_vector = torch.cat(all_action_masks, dim=0).float()
        num_actions = action_masks_vector.sum()

        # mean
        mean = (advantages_vector * action_masks_vector).sum() / (1e-10+num_actions)
        # std
        if not args.no_advantage_std_norm:
            var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / (1e-10+num_actions)
            rstd = var.clamp(min=1e-10).rsqrt()
        else:
            rstd = 1

        for exp in length_acc[length]:
            exp.advantages = (exp.advantages - mean) * rstd
            output_experiences.append(exp)
    return output_experiences

        

    