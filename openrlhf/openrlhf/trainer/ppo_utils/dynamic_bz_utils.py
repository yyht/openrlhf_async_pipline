
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import ray
import torch
from openrlhf.trainer.ppo_utils.seqlen_balancing import get_seqlen_balanced_partitions
from openrlhf.trainer.ppo_utils.seqlen_balancing_utils import get_seqlen_balanced_partitions as get_seqlen_balanced_partitions_packing_len
from openrlhf.utils.utils import zero_pad_sequences

class DynamicSamples:
    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    response_mask: Optional[torch.BoolTensor]

    sample_id: Optional[list[int]]

    @staticmethod
    def group_by_seqlen(samples_list, model_group, target_packing_length, pad_token_id):
        seqlen = [s.attention_mask.sum().tolist() for s in samples_list]
        num_actors = len(model_group._actor_handlers)
        k_partitions = num_actors // model_group.duplicate_actors
        group_partitions = get_seqlen_balanced_partitions(seqlen, k_partitions, equal_size=True)

        group_lens = [sum([seqlen[idx] for idx in group])/target_packing_length for group in group_partitions]
        dynamic_batch_size = int(mean(group_lens))+1

        dynamic_sample_list = []

        for group in group_partitions:
            group_seqlen = [seqlen[idx] for idx in group]
            dynamic_batch_group = get_seqlen_balanced_partitions_packing_len(group_seqlen, dynamic_batch_size, False, target_packing_length)
            samples_id.extend([[group[idx] for idx in dynamic_group] for dynamic_group in dynamic_batch_group])

            for dynamic_group in dynamic_batch_group:
                sequences = zero_pad_sequences([samples_list[group[idx]].sequences for idx in dynamic_group], side="right", value=pad_token_id)
                attention_mask = zero_pad_sequences([samples_list[group[idx]].attention_mask for idx in dynamic_group], side="right", value=0)
                action_mask = zero_pad_sequences([samples_list[group[idx]].action_mask for idx in dynamic_group], side="right", value=0)
                response_mask = zero_pad_sequences([samples_list[group[idx]].response_mask for idx in dynamic_group], side="right", value=0)
                sample_id = [group[idx] for idx in dynamic_group]
                dynamic_sample_list.append(DynamicSamples(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    response_mask=response_mask,
                    sample_id=sample_id
                ))
        return dynamic_sample_list



            






        
        
        
            
            
        
