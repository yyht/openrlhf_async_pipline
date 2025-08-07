import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import ray
import torch

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import PPORayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import zero_pad_sequences
from openrlhf.trainer.ppo_utils.advantage_utils import clip_by_abs
from openrlhf.trainer.ppo_utils.normalize_by_length import normalize_experience_by_length
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions

logger = init_logger(__name__)

def chunk_list(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        result.append(lst[i:i + chunk_size])
    return result


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    response_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None
    request_ids: Optional[list[str]] = None
    loss_mask: Optional[torch.Tensor] = None
    index: Optional[list[int]] = None
    rollout_log_probs: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.response_mask = to(self.response_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        self.loss_mask = to(self.loss_mask, device)
        self.rollout_log_probs = to(self.rollout_log_probs, device)
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.response_mask = pin_memory(self.response_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        self.loss_mask = pin_memory(self.loss_mask)
        self.rollout_log_probs = pin_memory(self.rollout_log_probs)
        return self


def split_experience_batch(experience: Experience) -> List[Experience]:
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
        "response_mask",
        "kl",
        "request_ids",
        "loss_mask",
        "rollout_log_probs"
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
            if isinstance(v, torch.Tensor):
                batch_kwargs[i][key] = v.unsqueeze(dim=0) # make a batch data
            else:
                batch_kwargs[i][key] = [v]

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
            batch_kwargs[i]["info"][k] = vv.unsqueeze(dim=0) # make a batch data

    items = [Experience(**kwargs) for kwargs in batch_kwargs]
    return items


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    response_mask: Optional[torch.BoolTensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]
    rewards: list[float]
    request_ids: list[str]
    index: Optional[list[int]]
    response: list[str]
    rollout_log_probs: Optional[torch.Tensor] = None

    def __init__(
        self,
        sequences=None,
        attention_mask=None,
        action_mask=None,
        response_mask=None,
        response_length=None,
        total_length=None,
        prompts=None,
        labels=None,
        rewards=None,
        packed_seq_lens=None,
        request_ids=None,
        index=None,
        response=None,
        rollout_log_probs=None
    ):
        self.sequences = sequences
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.response_mask = response_mask
        self.response_length = response_length
        self.total_length = total_length
        self.prompts = prompts or []
        self.labels = labels or []
        self.rewards = rewards or []
        self.packed_seq_lens = packed_seq_lens
        self.request_ids = request_ids or []
        self.index = index or []
        self.response = response or []
        self.rollout_log_probs = rollout_log_probs

    @staticmethod
    def concat_samples(samples_list: List["Samples"], pad_token_id) -> "Samples":
        """Concatenate multiple samples into one large sample.

        Args:
            samples_list: List of Samples to concatenate

        Returns:
            A new Samples instance containing all the concatenated data
        """
        if not samples_list:
            return Samples()

        # Concatenate tensor attributes with padding
        sequences = zero_pad_sequences([s.sequences for s in samples_list], side="right", value=pad_token_id)
        attention_mask = zero_pad_sequences([s.attention_mask for s in samples_list], side="right", value=0)
        action_mask = zero_pad_sequences([s.action_mask for s in samples_list], side="right", value=0)
        response_mask = zero_pad_sequences([s.response_mask for s in samples_list], side="right", value=0)
        if samples_list[0].rollout_log_probs is not None:
            rollout_log_probs = zero_pad_sequences([s.rollout_log_probs for s in samples_list], side="right", value=0)
        else:
            rollout_log_probs = None
        
        # Calculate response_length and total_length from masks
        response_length = response_mask.float().sum(dim=-1)
        total_length = attention_mask.float().sum(dim=-1)

        # Concatenate list attributes
        prompts = sum([s.prompts for s in samples_list], [])
        labels = sum([s.labels for s in samples_list], [])
        rewards = sum([s.rewards for s in samples_list], [])
        request_ids = sum([s.request_ids for s in samples_list], [])
        index = sum([s.index for s in samples_list], [])
        response = sum([s.response for s in samples_list], [])

        return Samples(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            response_mask=response_mask,
            response_length=response_length,
            total_length=total_length,
            prompts=prompts,
            labels=labels,
            rewards=rewards,
            packed_seq_lens=None,
            request_ids=request_ids,
            index=index,
            response=response,
            rollout_log_probs=rollout_log_probs
        )

def split_sample_batch(sample: Samples) -> List[Samples]:
    batch_size = len(sample.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = [
        'sequences',
        'attention_mask',
        'action_mask',
        'response_mask',
        'response_length',
        'total_length',
        'prompts',
        'labels',
        'rewards',
        'request_ids',
        'index',
        'response',
        'rollout_log_probs'
    ]

    for key in keys:
        value = getattr(sample, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            if isinstance(v, torch.Tensor):
                batch_kwargs[i][key] = v.unsqueeze(dim=0) # make a batch data
            else:
                batch_kwargs[i][key] = [v]
    items = [Samples(**kwargs) for kwargs in batch_kwargs]
    return items


class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        rollout_samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        return rollout_samples

        # tokenizer

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        llms = self.vllm_engines
        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id

        # Group outputs by micro_rollout_batch_size
        samples_list = []
        micro_rollout_batch_size = args.micro_rollout_batch_size
        for i in range(0, len(all_outputs), micro_rollout_batch_size):
            batch_outputs = all_outputs[i : i + micro_rollout_batch_size]
            batch_prompts = all_prompts[i : i + micro_rollout_batch_size]
            batch_labels = all_labels[i : i + micro_rollout_batch_size]

            # Calculate max lengths for this batch only
            batch_max_input_len = max(len(output.prompt_token_ids) for output in batch_outputs)
            batch_max_output_len = max(len(output.outputs[0].token_ids)+32 for output in batch_outputs)

            for batch_idx, output in enumerate(batch_outputs):

                # left padding input
                prompt_ids = list(output.prompt_token_ids)
                input_len = len(prompt_ids)
                input_ids = [pad_token_id] * (batch_max_input_len - input_len) + prompt_ids
                mask = [0] *  (batch_max_input_len - input_len) + [1] * input_len

                # right padding output
                output_ids = list(output.outputs[0].token_ids)
                if eos_token_id not in output_ids:
                    output_ids.append(eos_token_id)
                
                output_len = len(output_ids)
                output_ids = output_ids + [pad_token_id] * (batch_max_output_len - output_len)
                
                mask += [1] * output_len + [0] * (batch_max_output_len - output_len)
                env_mask = [1] * output_len + [0] * (batch_max_output_len - output_len)
                resp_mask = [1] * output_len + [0] * (batch_max_output_len - output_len)

                # concat input and output
                sequences = [input_ids + output_ids]
                attention_mask = [mask]
                action_mask = [env_mask]
                response_mask = [resp_mask]
            
                sequences = torch.tensor(sequences)
                attention_mask = torch.tensor(attention_mask)
                action_mask = torch.tensor(action_mask)
                response_mask = torch.tensor(response_mask)

                sequences = sequences.to("cpu")
                attention_mask = attention_mask.to("cpu")
                action_mask = action_mask.to("cpu")
                response_mask = response_mask.to("cpu")

                response_length = response_mask.float().sum(dim=-1)
                total_length = attention_mask.float().sum(dim=-1)

                rollout_samples = Samples(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    response_mask=response_mask,
                    response_length=response_length,
                    total_length=total_length,
                    prompts=batch_prompts[batch_idx:batch_idx+1],
                    labels=batch_labels[batch_idx:batch_idx+1],
                )
                samples_list.append(rollout_samples)

        return samples_list


class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        actor_model_group: PPORayActorGroup,
        critic_model_group: PPORayActorGroup,
        reward_model_group: PPORayActorGroup,
        initial_model_group: PPORayActorGroup,
        kl_controller,
        strategy=None,
        tokenizer=None,
        remote_reward_model=None,
        **kwargs,
    ):
        super().__init__()

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.advantage_estimator = strategy.args.advantage_estimator
        self.args = strategy.args

        # remote_rm_url indicates that the remote reward model is agent enviroment, remote http server or custom reward func
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_reward_model = remote_reward_model
        self.tokenizer = tokenizer

        # experience-queue
        self.experiences = []

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [int(i)]

        samples_list = []
        # if self.args.use_dynamic_batch:
        #     total_lengths = [int(s.total_length.item()) for s in rollout_samples]
        #     effective_actor_num = (
        #         self.args.actor_num_nodes
        #         * self.args.actor_num_gpus_per_node
        #         // self.args.ring_attn_size
        #         // self.args.ds_tensor_parallel_size
        #     )
        #     minimum_batch_num = get_minimum_num_micro_batch_size(
        #         total_lengths,
        #         self.args.max_tokens_per_gpu,
        #         self.args.ring_attn_size,
        #         self.args.ds_tensor_parallel_size,
        #     )
        #     minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
        #     num_batch = max(minimum_batch_num, effective_actor_num)
        #     batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
        #     for micro_index in batch_indexes:
        #         micro_batch = [rollout_samples[idx] for idx in micro_index]
        #         concat_samples = Samples.concat_samples(micro_batch, self.tokenizer.pad_token_id)
        #         samples_list.append(concat_samples)
        # else:
        batch_size = self.args.micro_rollout_batch_size
        for i in range(0, len(rollout_samples), batch_size):
            concat_samples = Samples.concat_samples(
                rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id,
            )
            samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Concat the samples into micro_rollout_batch_size
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        # TODO: balance the number of tokens of each batch for better performance
        # samples_list = []
        # batch_size = self.args.micro_rollout_batch_size
        # for i in range(0, len(rollout_samples), batch_size):
        #     concat_samples = Samples.concat_samples(rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id)
        #     samples_list.append(concat_samples)

        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)
        # experiences_request_ids = sum([exp.request_ids for exp in experiences], [])
        # for exp in experiences:
        #     exp.request_ids = None

        # # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences, samples_list

    @torch.no_grad()
    def make_experience(self, samples_list: List[Samples]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {len(samples_list[0].sequences) * len(samples_list)} samples")

        args = self.strategy.args
        device = "cpu"
        experiences = []

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]
        prompts_list = [p for s in samples_list for p in s.prompts]
        labels_list = [l for s in samples_list for l in s.labels]

        # Get rewards from samples, such as agent rewards
        rewards_list = [s.rewards for s in samples_list]
        r_refs = ray.put(rewards_list)

        # Get request-ids from samples
        request_ids_list = [s.request_ids for s in samples_list]
        request_ids_refs = ray.put(request_ids_list)

        logger.info({
            'INFO': '##MAKE-EXP##',
            'SAMPLE_SIZE': len(samples_list),
            'REWARD_SIZE': len(rewards_list),
            'D_TENSOR_SIZE': args.ds_tensor_parallel_size,
            'RING_ATTN_SIZE': args.ring_attn_size
        })

        # # Get rewards from samples, such as rewards from agent's environment
        # if samples_list[0].rewards:
        #     rewards_list = [s.rewards for s in samples_list]
        #     rewards_list = [torch.tensor(s.rewards) for s in samples_list]
        #     r_refs = ray.put(rewards_list)
        # elif self.remote_rm_url:
        #     queries_list = sum(
        #         [self.tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in sequences_list], []
        #     )
        #     prompts_list = sum([s.prompts for s in samples_list], [])
        #     labels_list = sum([s.labels for s in samples_list], [])
        #     r_refs = self.remote_reward_model.get_rewards.remote(queries_list, prompts_list, labels_list)
        # else:
        #     # Batch call reward model
        #     r_refs = self.reward_model_group.async_run_method_batch(
        #         method_name="forward",
        #         sequences=sequences_list,
        #         attention_mask=attention_mask_list,
        #         pad_sequence=[True] * len(samples_list),
        #     )

        # # Sync to avoid GPU OOM when colocate models
        # if args.colocate_all_models and not self.remote_rm_url:
        #     ray.get(r_refs)
        #     ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        # Batch call actor model
        action_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
            return_logits=[True] * len(sequences_list)
        )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Batch call critic model
        if self.critic_model_group is not None:
            # if args.colocate_critic_reward and not self.remote_rm_url:
            #     ray.get(r_refs)
            #     ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_ref = ray.put([[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size))

        # Batch call initial model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
                return_logits=[True] * len(sequences_list)
            )

            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(self.initial_model_group.async_run_method(method_name="empty_cache"))
        else:
            base_action_log_probs_ref = ray.put(
                [[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_dict_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        base_action_dict_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])
        rewards_list = ray.get(r_refs)
        request_ids_list = ray.get(request_ids_refs)

        logger.info({
            "INFO": "##AFTER-RAY-GATHER##",
            "VALUE": f"reward-size:{len(rewards_list)},request_ids-size:{len(request_ids_list)}",
        })

        assert (
            len(samples_list)
            == len(action_dict_list)
            == len(base_action_dict_list)
            == len(value_list)
            == len(rewards_list)
            == len(request_ids_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_dict_list): {len(action_dict_list)}, len(base_action_dict_list): {len(base_action_dict_list)}, len(value_list): {len(value_list)}, len(rewards_list): {len(rewards_list)}, len(request_ids_list)): {len(request_ids_list)}"

        # Process results for each sample
        for i, (samples, action_dict, base_action_dict, 
                value, rewards, request_ids) in enumerate(
            zip(samples_list, action_dict_list, 
            base_action_dict_list, value_list, rewards_list, request_ids_list)
        ):

            action_log_probs = action_dict['action_log_probs']
            base_action_log_probs = base_action_dict['action_log_probs']

            assert action_log_probs.shape == base_action_log_probs.shape
            assert action_log_probs.shape == samples.action_mask.shape

            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                kl_estimator=self.strategy.args.kl_estimator,
            )
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)

            sequences = samples.sequences
            attention_mask = samples.attention_mask

            rewards_tensor = torch.tensor([float(reward_info['rewards'][0]) for reward_info in rewards])
            
            # if args.use_ff_influence:
            rewards_tensor_score = torch.tensor([float(reward_info['answer_rewards'][0]) for reward_info in rewards])
            rewards_tensor_score = 2 * rewards_tensor_score - 1.0 # normalize to [-1, 1]
            action_mask = samples.action_mask
            # batch_size, 
            action_nll = masked_mean(-action_log_probs, action_mask, dim=-1) * rewards_tensor_score
            base_nll = masked_mean(-base_action_log_probs, action_mask, dim=-1) * rewards_tensor_score
            ff_influence = action_nll - base_nll
            
            action_ppl = (-masked_mean(action_log_probs, action_mask, dim=-1)).exp()
            base_ppl = (-masked_mean(base_action_log_probs, action_mask, dim=-1)).exp()

            action_self_certainty = action_dict['self_certainty']
            action_self_certainty = masked_mean(action_self_certainty, action_mask, dim=-1)

            base_self_certainty = base_action_dict['self_certainty']
            base_self_certainty = masked_mean(base_self_certainty, action_mask, dim=-1)

            base_entropy = masked_mean(base_action_dict['entropy'], action_mask, dim=-1)
            action_entropy = masked_mean(action_dict['entropy'], action_mask, dim=-1)

            info = {
                "kl": kl_mean,
                "reward": rewards_tensor,
                "response_length": samples.response_length,
                "total_length": samples.total_length,
                "base_entropy": base_entropy,
                "action_entropy": action_entropy,
                "ff_influence": ff_influence,
                "action_ppl": action_ppl,
                "base_ppl": base_ppl,
                "action_self_certainty": action_self_certainty,
                "base_self_certainty": base_self_certainty
            }

            monitor_rewards = {}
            for reward_info in rewards:
                for key in reward_info:
                    if not reward_info[key]:
                        continue
                    if key not in monitor_rewards:
                        monitor_rewards[key] = [float(reward_info[key][0])]
                    else:
                        monitor_rewards[key].append(float(reward_info[key][0]))
            for key in monitor_rewards:
                monitor_rewards[key] = torch.tensor(monitor_rewards[key])
            info.update(monitor_rewards)

            batch_size = sequences.shape[0]
            experience = Experience(
                sequences=sequences,
                action_log_probs=action_log_probs,
                base_action_log_probs=base_action_log_probs,
                values=value,
                returns=None,
                advantages=None,
                attention_mask=attention_mask,
                action_mask=samples.action_mask,
                response_mask=samples.response_mask,
                info=info,
                kl=kl,
                request_ids=request_ids,
                loss_mask=torch.ones(batch_size).float().to(sequences.device),
                index=samples.index,
                rollout_log_probs=samples.rollout_log_probs if samples.rollout_log_probs is not None else None
            )

            experiences.append(experience)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str} of total {len(experiences)} experiences")
        return experiences

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # get rewards from experiences
        # rewards = [experience.info["reward"] for experience in experiences]
        if self.args.use_dynamic_batch:
            # get rewards from experiences
            exp_len = [len(experience.index) for experience in experiences]
            # indices is an identity mapping when not using dynamic batch; otherwise, it maps back to the original indices after rearange samples
            indices = torch.tensor(sum([experience.index for experience in experiences], []))
            raw_rewards = torch.cat([experience.info[args.reward_key] for experience in experiences], dim=0)
            rewards = torch.empty_like(raw_rewards)
            rewards[indices] = raw_rewards  # sorted
        else:
            rewards = [experience.info[args.reward_key] for experience in experiences]
            rewards = torch.cat(rewards)

        # reward shaping
        if args.advantage_estimator == "rloo":
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            # rewards = rewards.reshape(-1).chunk(len(experiences))
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            rewards = rewards - rewards.mean(-1, keepdim=True)
            # rewards = rewards.reshape(-1).chunk(len(experiences))
        elif args.advantage_estimator in ["reinforce_baseline_nomean"]:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        elif args.advantage_estimator == "group_norm":
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            # rewards = rewards.reshape(-1).chunk(len(experiences))
        elif args.advantage_estimator == "opo":
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            response_length = [experience.info["response_length"] for experience in experiences]
            response_length = torch.cat(response_length).reshape(-1, args.n_samples_per_prompt)
            optimial_bias = (rewards * response_length).sum(dim=-1, keepdim=True) / (1e-10+response_length.sum(dim=-1, keepdim=True))
            rewards = rewards - optimial_bias
            # rewards = rewards.reshape(-1).chunk(len(experiences))
        elif args.advantage_estimator == "dpg":
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            kl = torch.cat([experience.info["kl"] for experience in experiences])
            response_length = torch.cat([experience.info["response_length"] for experience in experiences])
            kl *= response_length.to(kl.dtype) # log_actor - log_ref
            kl = kl.reshape(-1, args.n_samples_per_prompt)
            weight = -kl + rewards / args.dpg_beta
            Z = torch.logsumexp(weight, dim=-1, keepdim=True)
            Z += -torch.log(torch.tensor(float(args.n_samples_per_prompt)))
            rewards = (weight)-Z
            # rewards = rewards.reshape(-1).chunk(len(experiences))
        elif args.advantage_estimator == "reinforce_global":
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            rewards = rewards - rewards.mean(dim=None, keepdim=True)
            # rewards = rewards.reshape(-1).chunk(len(experiences))

        if self.args.use_dynamic_batch:
            rewards = rewards.reshape(-1)[indices].split(exp_len)
        else:
            rewards = rewards.reshape(-1).chunk(len(experiences))

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):

            if args.use_positive_gamma:
                answer_rewards = experience.info['answer_rewards']
                # gamma, [batch_size]
                gamma = (1.0-answer_rewards) * 1.0 + answer_rewards * args.gamma
                assert gamma.shape[0] == reward.shape[0]
            else:
                gamma = args.gamma

            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.response_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    gamma, # args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo", "opo", "dpg", "reinforce_global", "reinforce_baseline_nomean"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "reinforce_baseline_nomean",
                    "group_norm",
                    "dr_grpo",
                    "reinforce_global"
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    gamma, #args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None

        return experiences
    def normalize_experience(self,  experiences: List[Experience]):
        # Normalize advantages across all experiences
        if self.args.advantage_estimator in ["group_norm", "dr_grpo", "dpg"]:
            if self.args.advantage_clip_low_high:
                for exp in experiences:
                    exp.advantages = clip_by_abs(exp.advantages, clip_low=self.args.advantage_clip_low_high[0], clip_high=self.args.advantage_clip_low_high[1])
    
        if self.args.advantage_estimator not in ["group_norm", "dr_grpo", "dpg"]:

            if self.args.use_exp_norm_by_length:
                experiences = normalize_experience_by_length(experiences, self.args)
                logger.info({
                    'INFO': '##Normalize-by-LENGTH##',
                    'VALUE': 'NORMALIZEBYLENGTH'
                })
            else:
                all_advantages = []
                all_action_masks = []
                for exp in experiences:
                    if self.args.use_adv_mask_skip:
                        # skip loss-masked-item for global-adv-estimation
                        if exp.loss_mask.sum() < 1:
                            continue
                    all_advantages.append(exp.advantages.flatten())
                    all_action_masks.append(exp.action_mask.flatten())

                advantages_vector = torch.cat(all_advantages, dim=0).float()
                action_masks_vector = torch.cat(all_action_masks, dim=0).float()
                num_actions = action_masks_vector.sum()

                # mean
                mean = (advantages_vector * action_masks_vector).sum() / (1e-10+num_actions)
                # std
                if not self.args.no_advantage_std_norm:
                    var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / (1e-10+num_actions)
                    rstd = var.clamp(min=1e-10).rsqrt()
                else:
                    rstd = 1

                # Apply normalization to each experience
                for exp in experiences:
                    lower_bound = mean - 3 * var.sqrt()  # 下边界
                    upper_bound = mean + 3 * var.sqrt()
                    lower_bound_mask = (exp.advantages >= lower_bound).float()
                    upper_bound_mask = (exp.advantages <= upper_bound).float()
                    mask = lower_bound_mask * upper_bound_mask * exp.action_mask
                    within_range_ratio = mask.sum(dim=-1) / (exp.action_mask.sum(dim=-1)+1e-10)

                    bs = exp.advantages.shape[0]

                    exp.info["advantage_mean"] = torch.ones((bs,)) * mean
                    exp.info["advantage_std"] = torch.ones((bs,)) * var.sqrt()
                    exp.info["within_range_ratio"] = within_range_ratio

                    exp.advantages = (exp.advantages - mean) * rstd

                    if self.args.advantage_clip_low_high and self.args.advantage_estimator not in ['gae']:
                        exp.advantages = clip_by_abs(exp.advantages, clip_low=self.args.advantage_clip_low_high[0], clip_high=self.args.advantage_clip_low_high[1])

                logger.info({
                    'INFO': '##BATCH-NORMALIZATION-INFO##',
                    'MEAN': mean.item(),
                    'STD': var.sqrt().item(),
                })

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            if action_mask is not None:
                advantages_reversed.append(lastgaelam*action_mask[:, t])
            else:
                advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        
        if action_mask is not None:
            advantages *= action_mask
            returns *= action_mask
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return
            if action_mask is not None:
                returns[:, t] = action_mask[:, t] * returns[:, t]

        if action_mask is not None:
            returns = action_mask * returns

        return returns
