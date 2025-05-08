import random
import itertools, math, os, json
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .experience_maker import Experience

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    # add env-mask
    env_mask: Optional[torch.BoolTensor] = None


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "env_mask"
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], packing_samples=False) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "env_mask"
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask, env_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
            item.env_mask
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
            item.env_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
            env_mask[:right_pad] if item.env_mask is not None else None,
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, strategy, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []

        self.strategy = strategy
        self.gathered = False
        self.accuracy_lower_bound = float(1.0 / strategy.args.n_samples_per_prompt)
        self.accuracy_upper_bound = 0.8

        self.left_items: List[BufferItem] = []
        self.rollout_step = 0
        self.is_normalized = False
        self.failed_rollout_step = 0

        logger.info({
            'INFO': 'REPLAY_BUFFER_CPU_OFFLOAD',
            'VALUE': self.cpu_offload
        })

    def all_gather(self, steps=None):

        if self.strategy.is_rank_0():
            logger.info({
                "INFO": "FILTER_BEFORE",
                "ITEMSIZE": len(self.items),
                "LEFTSIZE": len(self.left_items),
                'IS_RANK_0': self.strategy.is_rank_0()
            })

        cutoff = len(self.items) % self.strategy.args.n_samples_per_prompt
        if self.strategy.args.use_acc_filter:
            if cutoff == 0:
                self.filter_acc()
            if self.strategy.is_rank_0():
                logger.info({
                    "INFO": "ACC_FILTER_AFTER",
                    "ITEMSIZE": len(self.items),
                    "LEFTSIZE": len(self.left_items),
                    'IS_RANK_0': self.strategy.is_rank_0()
                })
        
        if self.strategy.args.use_length_filter:
            self.filter_seq_length()
            if self.strategy.is_rank_0():
                logger.info({
                    "INFO": "LENGTH_FILTER_AFTER",
                    "ITEMSIZE": len(self.items),
                    "LEFTSIZE": len(self.left_items),
                    'IS_RANK_0': self.strategy.is_rank_0()
                })

        if self.strategy.args.use_eval_filter:
            failed_items = self.filter_eval_fail()
            if self.strategy.is_rank_0():
                logger.info({
                    "INFO": "EVAL_FAIL_FILTER_AFTER",
                    "ITEMSIZE": len(self.items),
                    "LEFTSIZE": len(self.left_items),
                    'IS_RANK_0': self.strategy.is_rank_0(),
                    'FAILED_ITEMS': len(failed_items)
                })

            if failed_items:
                actor_rank = self.strategy.get_rank()
                failed_output_path = os.path.join(self.strategy.args.save_path, f'failed_sample_steps_{steps}_rank_{actor_rank}.jsonl')
                if os.path.exists(failed_output_path):
                    failed_output_path = os.path.join(self.strategy.args.save_path, f'failed_sample_steps_{steps}_{self.failed_rollout_step}_rank_{actor_rank}.jsonl')
                    self.failed_rollout_step += 1
                with open(failed_output_path, 'w') as fwobj:
                    for item in failed_items:
                        tmp = {
                            'info': {}
                        }
                        for key in item.info:
                            if isinstance(item.info[key], torch.Tensor):
                                tmp['info'][key] = item.info[key].numpy().item()
                            else:
                                tmp['info'][key] = item.info[key]
                        tmp['sequences'] = item.sequences.numpy().tolist()
                        tmp['action_log_probs'] = item.action_log_probs.numpy().tolist()
                        fwobj.write(json.dumps(tmp, ensure_ascii=False)+'\n')

        if self.strategy.args.reuse_offline:
            if self.strategy.is_rank_0():
                self.items.extend(self.left_items)
                self.left_items.clear()
            if self.strategy.is_rank_0():
                logger.info({
                    "INFO": "REUSE_OFFLINE",
                    "ITEMSIZE": len(self.items),
                    "LEFTSIZE": len(self.left_items),
                    'IS_RANK_0': self.strategy.is_rank_0()
                })

        chunk_size = math.ceil(len(self.items) / self.strategy.args.n_samples_per_prompt)
        gathered_data = []
        gathered_failed_data = []

        for i in range(self.strategy.args.n_samples_per_prompt):
            chunk = self.items[i * chunk_size : (i + 1) * chunk_size]
            all_chunks: list[list[BufferItem]] = [None] * dist.get_world_size()
            dist.all_gather_object(all_chunks, chunk)
            chunk_data = [
                item.to_device(self.target_device) if not self.cpu_offload else item
                for item in itertools.chain.from_iterable(all_chunks)
            ]
            gathered_data.extend(chunk_data)
        
        self.items = gathered_data

        self.gathered = True

        if self.strategy.is_rank_0():
            logger.info({
                    "INFO": "ALLGATHERBEFORE",
                    "ITEMSIZE": len(self.items),
                    "LEFTSIZE": len(self.left_items) 
                })
            if self.strategy.args.dump_data:
                output_path = os.path.join(self.strategy.args.save_path, f'rollout_sample_steps_{steps}.jsonl')
                if os.path.exists(output_path):
                    output_path = os.path.join(self.strategy.args.save_path, f'rollout_sample_steps_{steps}_{self.rollout_step}.jsonl')
                    self.rollout_step += 1
                with open(output_path, 'w') as fwobj:
                    for item in self.items:
                        tmp = {
                            'info': {}
                        }
                        for key in item.info:
                            if isinstance(item.info[key], torch.Tensor):
                                tmp['info'][key] = item.info[key].numpy().item()
                            else:
                                tmp['info'][key] = item.info[key]
                        tmp['sequences'] = item.sequences.numpy().tolist()
                        tmp['action_log_probs'] = item.action_log_probs.numpy().tolist()
                        fwobj.write(json.dumps(tmp, ensure_ascii=False)+'\n')

        cutoff = len(self.items) % self.strategy.args.train_batch_size
        if cutoff != 0:
            cutoff_items = self.items[:-cutoff]
            if len(cutoff_items) > 0:
                if self.strategy.args.reuse_offline:
                    self.left_items.extend(self.items[-cutoff:])
                self.items = cutoff_items
            else:
                self.left_items.extend(self.items)

            logger.info({
                "INFO": "ALLGATHER",
                "CUTOFF": cutoff,
                "ITEMSIZE": len(self.items),
                "LEFTITEMSIZE": len(self.left_items)
            })

    @torch.no_grad()
    def filter_acc(self):
        rewards = torch.stack([torch.tensor(item.info['answer_rewards']) for item in self.items])
        reward_matrix = (rewards==1.0).reshape(-1, self.strategy.args.n_samples_per_prompt).float()
        acc_tensor = torch.mean(reward_matrix, dim=-1)
        acc_mask = (acc_tensor >= self.accuracy_lower_bound) & (
	                acc_tensor <= self.accuracy_upper_bound)
        batch_acc_mask = acc_mask.repeat_interleave(self.strategy.args.n_samples_per_prompt)
        batch_acc_tensor = acc_tensor.repeat_interleave(self.strategy.args.n_samples_per_prompt)
        valid_items = []
        for acc_mask, acc_tesnor, item in zip(batch_acc_mask, batch_acc_tensor, self.items):
            if self.strategy.args.sft_loss_coef > 0:
                acc_ratio = acc_tesnor.item()
                if acc_ratio <= 2 * self.accuracy_lower_bound and item.info['answer_rewards'] == 1.0:
                    item.info['acc_ratio'] = 1.0
                else:
                    item.info['acc_ratio'] = 0.0
            if acc_mask:
                valid_items.append(item)
        self.items = valid_items

    @torch.no_grad()
    def filter_eval_fail(self):
        rule_fail_mask = torch.tensor([item.info['rule_eval_fails'] for item in self.items])
        model_fail_mask = torch.tensor([item.info['model_eval_fails'] for item in self.items])
        repeatness_mask = torch.tensor([item.info['repeatness'] for item in self.items])
        valid_items = []
        failed_items = []
        for rule_fail, model_fail, repeatness, item in zip(rule_fail_mask, model_fail_mask, repeatness_mask, self.items):
            if rule_fail > 0 or model_fail > 0:
                failed_items.append(item)
                continue
            if repeatness > 0.05: # need to check repeatness on r1-sft-dataset
                continue
            valid_items.append(item)
        self.items = valid_items
        return failed_items

    @torch.no_grad()
    def filter_seq_length(self):
        max_output_len = self.strategy.args.generate_max_len
        filtered_index = []
        for idx, item in enumerate(self.items):
            if item.info['response_length'] > max_output_len:
                continue
            filtered_index.append(idx)
        self.items = [self.items[idx] for idx in filtered_index]

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.gathered = False
        self.is_normalized = False
        self.items.clear()
        if self.strategy.args.reuse_offline:
            if not self.strategy.is_rank_0():
                self.left_items.clear()
        logger.info({
            "INFO": "CLEAR",
            "ITEMSIZE": len(self.items),
            "LEFTSIZE": len(self.left_items),
            'IS_RANK_0': self.strategy.is_rank_0()
        })

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            if item.env_mask is not None:
                action_masks.append(item.env_mask)
            else:
                action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # For Data Parallel (DP)
        # Calculate mean
        sum_and_count = (items_vector.sum(), num_actions)
        all_sum, all_count = (
            strategy.all_reduce(torch.tensor(sum_and_count, device=items_vector.device), "sum")
            if not self.gathered
            else sum_and_count
        )
        mean = all_sum / all_count
        # Calculate standard deviation
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum") if not self.gathered else std
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
        self.is_normalized = True

    def normalize_mean(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            if item.env_mask is not None:
                action_masks.append(item.env_mask)
            else:
                action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # For Data Parallel (DP)
        # Calculate mean
        sum_and_count = (items_vector.sum(), num_actions)
        all_sum, all_count = (
            strategy.all_reduce(torch.tensor(sum_and_count, device=items_vector.device), "sum")
            if not self.gathered
            else sum_and_count
        )
        mean = all_sum / all_count

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean))