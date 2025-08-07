from typing import List

import ray
import torch

from openrlhf.trainer.ppo_utils.experience_maker import Samples, SamplesGenerator
from openrlhf.utils.logging_utils import init_logger
logger = init_logger(__name__)

class SamplesGeneratorAsync(SamplesGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            stop=["User:", "Human:", "Assistant:", "</answer>", self.tokenizer.eos_token],
            logprobs=1 if args.enable_vllm_is_correction else None,
        )
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
            labels = all_labels[i * batch_size : (i + 1) * batch_size]
            prompt_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]

            refs.append(
                llm.add_requests.remote(
                    sampling_params=sampling_params,
                    prompts=prompts,
                    prompt_ids=prompt_ids,
                    labels=labels,
                    max_length=truncate_length,
                    hf_tokenizer=self.tokenizer,
                )
            )
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        # Group outputs by prompt
        prompt_groups = {}
        for output in all_outputs:
            request_id = output.request_id.split('####idx:')[0]
            prompt_groups.setdefault(request_id, []).append(output)

        logger.info({
            'INFO': '##SIZE-OF-GROUPS##',
            'VALUE': f"size of groups {len(prompt_groups)} and prompts {len(all_prompts)} and n_samples_per_prompt {n_samples_per_prompt}"
        })

        # Reorder outputs to keep same prompts together
        # This is very important for REINFORCE++-baseline/GRPO/RLOO
        all_outputs = []
        for prompt in prompt_groups.keys():
            if len(prompt_groups[prompt]) < n_samples_per_prompt:
                continue
            all_outputs.extend(prompt_groups[prompt])

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id

        # Group outputs by micro_rollout_batch_size
        samples_list = []
        micro_rollout_batch_size = args.micro_rollout_batch_size
        for i in range(0, len(all_outputs), micro_rollout_batch_size):
            batch_outputs = all_outputs[i : i + micro_rollout_batch_size]
            batch_rewards = [output.outputs[0].reward_info for output in batch_outputs]
            batch_request_ids = [output.request_id for output in batch_outputs]
            batch_labels = [output.label for output in batch_outputs]
            batch_prompts = [output.prompt for output in batch_outputs]
            batch_responses = [output.outputs[0].text for output in batch_outputs]
            
            # Calculate max lengths for this batch only
            batch_max_input_len = max(len(output.prompt_token_ids) for output in batch_outputs)
            batch_max_output_len = max(len(output.outputs[0].token_ids)+32 for output in batch_outputs)

            for batch_idx, output in enumerate(batch_outputs):
                # Tokenize state
                assert len(output.outputs[0].token_ids) == len(output.outputs[0].action_mask)

                # left padding input
                prompt_ids = list(output.prompt_token_ids)
                input_len = len(prompt_ids)
                input_ids = [pad_token_id] * (batch_max_input_len - input_len) + prompt_ids
                mask = [0] *  (batch_max_input_len - input_len) + [1] * input_len

                # right padding output
                output_ids = list(output.outputs[0].token_ids)
                output_len = len(output_ids)
                output_ids = output_ids + [pad_token_id] * (batch_max_output_len - output_len)

                log_probs = output.outputs[0].log_probs
                if log_probs is not None:
                    log_probs = list(log_probs)
                    log_probs = log_probs + [0.0] * (batch_max_output_len - output_len)

                mask += [1] * output_len + [0] * (batch_max_output_len - output_len)
                env_action_mask = list(output.outputs[0].action_mask) + [0] * (batch_max_output_len - len(output.outputs[0].action_mask))
                trajectory_mask = [1] * output_len + [0] * (batch_max_output_len - output_len)

                assert len(env_action_mask) == len(trajectory_mask)

                # concat input and output
                sequences = [input_ids + output_ids]
                attention_mask = [mask]
                action_mask = [env_action_mask]
                response_mask = [trajectory_mask]
                
                if log_probs is not None:
                    log_probs = [log_probs]
                    log_probs = torch.tensor(log_probs)
                    log_probs = log_probs.to("cpu")
                
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
                    rewards=batch_rewards[batch_idx:batch_idx+1],
                    request_ids=batch_request_ids[batch_idx:batch_idx+1],
                    response=batch_responses[batch_idx:batch_idx+1],
                    rollout_log_probs=log_probs
                )
                samples_list.append(rollout_samples)

        return samples_list
