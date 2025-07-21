import asyncio
import os
import time

import os
import time
from abc import ABC
from datetime import timedelta
import random
import numpy as np
from typing import Optional, Any, List, Dict, Tuple

import ray
import torch
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ray.launcher import PPORayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.trainer.ppo_utils.experience_maker import split_experience_batch, split_sample_batch
from openrlhf.trainer.ppo_utils.seqlen_balancing import get_seqlen_balanced_partitions
from openrlhf.trainer.ppo_utils.exp_balancing import balanced_subset
from openrlhf.env.filter_config import FILTER_FN_CONFIG
from openrlhf.trainer.ppo_utils.exp_statistics import sample_statistics, sample_strategy, length_filter, length_balance
from openrlhf.trainer.ppo_utils.experience_maker import Samples, Experience
from openrlhf.trainer.ppo_utils.outlier_utils import detect_outliers_iqr
from openrlhf.trainer.ppo_utils.mopps_utils import MoPPS
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.utils.logging_utils import init_logger
logger = init_logger(__name__)

def dump_samples(args, eposide, samples, prefix='rollout_sample_eposide'):
    import os, json
    output_path = os.path.join(args.save_path, f'{prefix}_{eposide}.jsonl')
    logger.info(f'Dump samples to {output_path}')
    with open(output_path, 'w') as fwobj:
        for item in samples:
            tmp = item.__dict__.copy()
            for key in tmp:
                if isinstance(tmp[key], torch.Tensor):
                    if tmp[key].dtype == torch.int64:
                        tmp[key] = tmp[key].int().numpy()
                    else:
                        tmp[key] = tmp[key].numpy()
                    tmp[key] = tmp[key].tolist()
            fwobj.write(json.dumps(tmp, ensure_ascii=False)+'\n')

def dump_experiences(args, eposide, experiences, prefix='train_eposide'):
    import os, json
    output_path = os.path.join(args.save_path, f'{prefix}_{eposide}.jsonl')
    logger.info(f'Dump samples to {output_path}')
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
        "loss_mask"
    )

    with open(output_path, 'w') as fwobj:
        for experience in experiences:
            tmp = {}
            for key in keys:
                value = getattr(experience, key)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.int64:
                        vals = value.int().numpy().tolist()
                    else:
                        vals = value.numpy().tolist()
                else:
                    vals = value
                tmp[key] = vals
            tmp['info'] = {}
            for key, value in experience.info.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.int64:
                        tmp['info'][key] = value.int().numpy().tolist()
                    else:
                        tmp['info'][key] = value.numpy().tolist()
                else:
                    tmp['info'][key] = value
            fwobj.write(json.dumps(tmp, ensure_ascii=False)+'\n')

@ray.remote(num_cpus=0)
class SignalActor:
    def __init__(self):
        self.generating_event = asyncio.Event()
        self.update_weights_event = asyncio.Event()
        self.generating_event.set()  # Initially allow generation
        self.update_weights_event.set()  # Initially allow weight updates

        self.make_experience_event = asyncio.Event()
        self.make_experience_event.set()

        self.train_event = asyncio.Event()
        self.train_event.set()

    async def wait_generating(self):
        """Wait for generation to be allowed."""
        return await self.generating_event.wait()

    async def wait_update_weights(self):
        """Wait for weight update to be allowed."""
        return await self.update_weights_event.wait()

    async def wait_make_experience(self):
        """Wait for make_experience to be allowed."""
        return await self.make_experience_event.wait()

    async def wait_train(self):
        """Wait for make_experience to be allowed."""
        return await self.train_event.wait()

    def set_generating(self, allow_generating):
        """Set generation state.

        Args:
            is_generating: True to allow generation, False to block it
        """
        if allow_generating:
            self.generating_event.set()
        else:
            self.generating_event.clear()

    def set_update_weights(self, allow_updating):
        """Set weight update state.

        Args:
            is_updating: True to allow weight updates, False to block it
        """
        if allow_updating:
            self.update_weights_event.set()
        else:
            self.update_weights_event.clear()

    def set_make_experience(self, allow_making):
        if allow_making:
            self.make_experience_event.set()
        else:
            self.make_experience_event.clear()

    def set_train(self, allow_train):
        if allow_train:
            self.train_event.set()
        else:
            self.train_event.clear()


from pydantic import BaseModel
class QueueData(BaseModel):
    samples: Optional[List[Any]] = None
    experiences: Optional[List[Any]] = None
    data_loader_state_dict: Optional[Any] = None
    info: Optional[Dict] = None
    is_done: Optional[bool] = False

class MetaData(BaseModel):
    steps: Optional[int] = 0



@ray.remote
class GenerateSamplesActor(BasePPOTrainer):
    def __init__(self, *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        super().__init__(*args, **kwargs)

        self.samples_generator = self.generator_cls(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )
        self.prepare_datasets()

        self.prompts_code_exec_dict = {}

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=None,
        )

        if self.args.use_mopps:
            self.mopps_api = MoPPS(
                alpha_prior=1.0,
                beta_prior=1.0,
                exp_ratio=0.5,
                target_success_rate=0.5,
                top_k_ratio=0.5)

    def generate_samples(self, prompts, labels, **generate_kwargs):
        return self.samples_generator.generate_samples(prompts, labels, **generate_kwargs)

    def fit(self, start_episode, queue, data_loader_state_dict, train_meta_queue):
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)
            logger.info({
                'INFO': '##LOAD-DATA-LOADER-STATE-DICT##',
                'VALUE': data_loader_state_dict
            })
            start_idx = data_loader_state_dict['_num_yielded']
        else:
            start_idx = 0

        first_count = 0
        ideal_buffer_size = self.args.n_samples_per_prompt * self.args.rollout_batch_size
        consumed_sample_nums = 0
        # is_generated_full = True
        for episode in range(start_episode, self.args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Generate Episode [{episode + 1}/{self.args.num_episodes}]",
                disable=False,
            )

            filtered_samples = []
            number_of_samples = 0
            train_steps = 0
            queue_info = {}
            for _, rand_prompts, labels in (self.prompts_dataloader):

                if self.args.use_mopps:
                    prompts_uuids = {}
                    for label in labels:
                        meta_info = json.loads(label)
                        prompts_uuids[meta_info["uuid"]] = []
                    prompt_indices = self.mopps_api.sample(prompts_uuids)
                    rollout_prompts = [rand_prompts[indice] for indice in prompt_indices]
                    rollout_labels = [labels[indice] for indice in prompt_indices]
                else:
                    rollout_prompts = rand_prompts
                    rollout_labels = labels

                # actual_buffer_size = ideal_buffer_size - max([0, consumed_sample_nums])
                # if actual_buffer_size <= 0:
                #     consumed_sample_nums = 0
                #     is_generated_full = False
 
                # Wait until queue is not full
                # To support 1-step off-policy training
                while queue.full():
                    if self.args.use_exp_before_queue:
                        ray.get(self.signal_actor.set_train.remote(True))
                    logger.info(f"Queue is full, waiting for training to consume samples ${queue.qsize()}...")
                    time.sleep(1)  # Wait for 1 second before checking again
                    consumed_sample_nums = 0

                # Wait for generation to be allowed
                ray.get(self.signal_actor.wait_generating.remote())
                # Block weight updates
                ray.get(self.signal_actor.set_update_weights.remote(False))

                if self.args.use_exp_before_queue:
                    ray.get(self.signal_actor.set_train.remote(False))

                # Generate samples
                # remote_reward_model is used to get rewards for dynamic sampling
                rollout_samples = self.generate_samples(
                    rollout_prompts, rollout_labels, **self.generate_kwargs
                )

                # Dump rollout samples
                dump_idx = episode * self.prompts_dataloader.__len__() + start_idx
                dump_samples(self.args, dump_idx, rollout_samples, prefix='rollout_sample_eposide')

                start_idx += 1
                # Allow weight updates after generation is done
                ray.get(self.signal_actor.set_update_weights.remote(True))

                pbar.update()

                # dynamic filtering
                pass_rate = 1.0
                filtered_samples_nums = 0.0
                format_samples_nums = 0.0
                if self.args.dynamic_filtering:

                    from collections import Counter
                    filter_counter = Counter()
                    number_of_samples += len(rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        task = set([sample.labels[0]['task'].lower() for sample in batch_samples])
                        assert len(task) == 1
                        task = list(task)[0]

                        sample_statistics = {}
                        for reward_key in batch_samples[0].rewards[0]:
                            # reward_mean = np.mean([sample.rewards[0][reward_key][0] for sample in batch_samples])
                            # reward_std = np.std([sample.rewards[0][reward_key][0] for sample in batch_samples])
                            # sample_statistics[f'{reward_key}_mean'] = reward_mean
                            # sample_statistics[f'{reward_key}_std'] = reward_std
                            sample_statistics[f'{reward_key}_max'] = float(np.max([sample.rewards[0][reward_key][0] for sample in batch_samples]))
                            sample_statistics[f'{reward_key}_min'] = float(np.min([sample.rewards[0][reward_key][0] for sample in batch_samples]))

                        for sample in batch_samples:
                            for key in sample_statistics:
                                sample.rewards[0][key] = [sample_statistics[key]]
                            sample.rewards[0]['is_valid'] = [1.0]
                            sample.rewards[0]['is_filter'] = [0.0]

                        if self.args.use_outlier_mask:
                            outliers_idx = {}
                            key_list = ['repeatness', 'repetition_penalty']
                            for key in key_list:
                                outlier_score = [sample.rewards[0][key][0] for sample in batch_samples]
                                outliers_idx[key] = detect_outliers_iqr(outlier_score)
                            common_outliers =  set(outliers_idx[key_list[0]])
                            for key in outliers_idx:
                                common_outliers.intersection_update(outliers_idx[key])

                            # 转换为列表（如果需要）
                            common_outliers_idx = list(common_outliers)

                            for sample_idx, sample in enumerate(batch_samples):
                                if sample_idx in common_outliers_idx:
                                    sample.rewards[0]['is_valid'] = [0.0]

                        if self.args.use_filter_sample:
                            avg_reward_filter_ratio = 0
                            for sample in batch_samples:
                                flag = FILTER_FN_CONFIG[f'{task}_exp_filter_fn'](None, sample, args=self.args)
                                if not flag and sample.rewards[0]['answer_rewards'][0] > 0.5:
                                    avg_reward_filter_ratio += 1
                                if flag:
                                    sample.rewards[0]['is_filter'] = [1.0]

                        # Calculate average reward for this batch of samples
                        avg_reward = sum([sample.rewards[0]['answer_rewards'][0] for sample in batch_samples]) / len(batch_samples)
                        format_reward = sum([sample.rewards[0]['format_answer'][0] for sample in batch_samples]) / len(batch_samples)
                        quality_score = sum([FILTER_FN_CONFIG[f'{task}_sample_filter_fn'](sample) for sample in batch_samples])
                        invalid_score = sum([FILTER_FN_CONFIG[f'{task}_reward_fail_fn'](sample) for sample in batch_samples]) / len(batch_samples)

                        if self.args.use_mopps:
                            sample_uuids = batch_samples[0].labels[0]['uuid']
                            prompts_uuids[sample_uuids] = [sample.rewards[0]['answer_rewards'][0] for sample in batch_samples]

                        if train_meta_queue.full():
                            train_meta_data = train_meta_queue.get()
                            train_steps = train_meta_data.steps

                        if format_reward > 0.5:
                            filter_counter['format_cnt'] += 1

                        if invalid_score > 0.4:
                            filter_counter['invalid_cnt'] += 1
                        
                        # Check if average reward is within the specified range
                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            if invalid_score > 0.4:
                                continue
                            filtered_samples.extend(batch_samples)

                    if self.args.use_length_acc_balance:
                        before_filter_sample_nums = len(filtered_samples)
                        filtered_samples = length_balance(filtered_samples)
                        after_filter_sample_nums = len(filtered_samples)
                        filtered_samples_nums = after_filter_sample_nums - before_filter_sample_nums

                    logger.info(
                        f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling with {filter_counter}"
                    )

                    num_actors = len(self.actor_model_group._actor_handlers)
                    effective_actors = num_actors // self.actor_model_group.duplicate_actors
                    chunk_size = len(filtered_samples) // (effective_actors * self.args.micro_rollout_batch_size)

                    if chunk_size <= 4:
                        continue

                    sample_size = chunk_size * effective_actors * self.args.micro_rollout_batch_size
                    sample_chunk_size = sample_size // self.args.n_samples_per_prompt
                    actual_sample_size = sample_chunk_size * self.args.n_samples_per_prompt
                    
                    rollout_samples = filtered_samples[:actual_sample_size]
                    pass_rate = len(filtered_samples) / number_of_samples * 100

                    logger.info(
                        f"rollout_samples {len(rollout_samples)} left_samples {len(filtered_samples[actual_sample_size:])} consumed_samples: {consumed_sample_nums}"
                    )

                    number_of_samples = 0
                    filtered_samples = []

                    if self.args.use_mopps:
                        self.mopps_api.update(prompts_uuids)
                    
                # Dump make-exp samples
                dump_samples(self.args, dump_idx, rollout_samples, prefix='make_exp_eposide')

                if self.args.use_exp_before_queue:
                    # Wait for generation to be allowed
                    experiences, samples = self.experience_maker.make_experience_batch(rollout_samples)
                else:
                    experiences = [None]*len(rollout_samples)
                    samples = rollout_samples

                queue_data = QueueData(
                    samples=samples,
                    experiences=experiences,
                    data_loader_state_dict=self.prompts_dataloader.state_dict(),
                    info={
                        'pass_rate': pass_rate,
                        'dump_idx': dump_idx,
                        'filtered_samples_nums': filtered_samples_nums,
                        'episode': episode
                    },
                    is_done=False
                )
                queue.put(queue_data)

        queue_data = QueueData(
            samples=None,
            experiences=None,
            data_loader_state_dict=self.prompts_dataloader.state_dict(),
            info={},
            is_done=True
        )
        queue.put(queue_data)

@ray.remote
class TrainingActor(BasePPOTrainer):
    def __init__(self, *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        self.remote_reward_model = kwargs.pop("remote_reward_model")
        super().__init__(*args, **kwargs)

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=None,
        )

        self._init_wandb()
        self.eval_dataloader = None

        self.experiences = []
        self.samples = []
        self.samples_info_list = []

        self.old_experiences = []
        self.old_samples_info_list = []

    def _broadcast_to_vllm(self):
        if self.vllm_engines is not None:
            # Block generation
            ray.get(self.signal_actor.set_generating.remote(False))
            # Wait for weight updates to be allowed
            ray.get(self.signal_actor.wait_update_weights.remote())

            # Perform weight update
            ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

            # Allow generation
            ray.get(self.signal_actor.set_generating.remote(True))

    def fit(self, queue, steps, train_meta_queue):
        args = self.args

        is_train = False

        while True:

            train_meta_data = MetaData(
                steps=steps
            )
            train_meta_queue.put(train_meta_data)

            queue_data = queue.get()

            if queue_data.is_done:
                break

            queue_info = queue_data.info
            dump_idx = queue_info['dump_idx']
            pass_rate = queue_info['pass_rate']
            filtered_samples_nums = queue_info['filtered_samples_nums']
            episode = queue_info['episode']
            data_loader_state_dict = queue_data.data_loader_state_dict

            if not self.args.use_exp_before_queue:
                experiences, samples = self.experience_maker.make_experience_batch(queue_data.samples)
            else:
                samples = queue_data.samples
                experiences = queue_data.experiences

            # # filter-experiences
            experience_num = self.args.rollout_batch_size * self.args.n_samples_per_prompt
            assert len(experiences) == len(samples)
            for experience_batch, samples_batch in zip(experiences, samples):
                experience_list = split_experience_batch(experience_batch)
                sample_list = split_sample_batch(samples_batch)
                assert len(experience_list) == len(sample_list)
                for experience, sample in zip(experience_list, sample_list):
                    task = sample.labels[0]['task'].lower()
                    if self.args.use_filter_sample:
                        flag = FILTER_FN_CONFIG[f'{task}_exp_filter_fn'](experience, sample, args=self.args)
                        if flag or sample.rewards[0]['is_valid'][0] < 1:
                            # adv-mask-before
                            if self.args.use_loss_mask and not self.args.use_adv_mask_before:
                                experience.loss_mask = torch.zeros_like(experience.loss_mask)
                            else:
                                continue
                    self.experiences.append(experience)
                    self.samples_info_list.append(sample)

            logger.info(
                        f"filtered_experiences {len(self.experiences) / experience_num} < rollout_batch_size {experience_num}, continue sampling"
            )

            if is_train:
                is_train = False
                step_behind_samples = len(self.experiences)

            if len(self.experiences)+len(self.old_experiences) < experience_num:
                continue

            experiences = self.experiences
            samples_info_list = self.samples_info_list

            logger.info({
                'INFO': '##EXPRIENCES_INFO##',
                'VALUE': f"EXP: {len(experiences)},SAMPLE: {len(samples_info_list)}"
            })

            self.experiences = []
            self.samples_info_list = []

            dump_experiences(self.args, steps, experiences, prefix=f'train_eposide_experiences_{dump_idx}')
            # Dump make-exp samples
            dump_samples(self.args, steps, samples_info_list, prefix=f'train_eposide_samples_{dump_idx}')

            if not self.args.remove_advantage_whiten:
                # Breaking Habits: On the Role of the Advantage Function in Learning Causal State Representations
                experiences = self.experience_maker.normalize_experience(experiences)

            actual_experiences = []
            for experience, sample in zip(experiences, samples_info_list):
                task = sample.labels[0]['task'].lower()
                if self.args.use_filter_sample:
                    flag = FILTER_FN_CONFIG[f'{task}_exp_filter_fn'](experience, sample, args=self.args)
                    if flag or sample.rewards[0]['is_valid'][0] < 1:
                        if self.args.use_loss_mask and not self.args.use_adv_mask_after:
                            experience.loss_mask = torch.zeros_like(experience.loss_mask)
                        else:
                            continue
                    actual_experiences.append(experience)

            logger.info({
                'INFO': '##SKIP_EXPRIENCES##',
                'VALUE': f"BEFORE: {len(experiences)},AFTER: {len(actual_experiences)}"
            })

            experiences = actual_experiences
                    
            sample0 = self.tokenizer.batch_decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)
            print(sample0)
            if self.args.use_dynamic_bs_ds or self.args.use_dynamic_batch:
                num_actors = len(self.actor_model_group._actor_handlers)
                effective_actors = num_actors // self.actor_model_group.duplicate_actors
                chunk_size = len(experiences) // (effective_actors * self.args.micro_train_batch_size)

                sample_size = chunk_size * effective_actors * self.args.micro_train_batch_size
                sample_chunk_size = sample_size // self.args.n_samples_per_prompt
                actual_sample_size = sample_chunk_size * self.args.n_samples_per_prompt
                actor_experiences = experiences[:actual_sample_size]

                logger.info({
                    'INFO': '##DYNAMIC_BATCH_SIZE_ACTOR##',
                    'VALUE': len(actor_experiences)
                })
            else:
                actor_experiences = experiences

            if self.args.use_seq_balancing:
                seqlen = [exp.attention_mask.sum().tolist() for exp in actor_experiences]
                num_actors = len(self.actor_model_group._actor_handlers)
                k_partitions = num_actors // self.actor_model_group.duplicate_actors
                actor_experience_ids = get_seqlen_balanced_partitions(seqlen, k_partitions, equal_size=True)
                experiences_actor = []
                for group_ids in actor_experience_ids:
                    for idx in group_ids:
                        experiences_actor.append(actor_experiences[idx])
            else:
                experiences_actor = actor_experiences

            # balance experiences across dp
            if args.use_dynamic_batch:
                experiences_actor = balance_experiences(experiences_actor, args, mode='actor')

            refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences_actor)
            if self.critic_model_group is not None:
                if self.args.use_dynamic_bs_ds or self.args.use_dynamic_batch:
                    num_actors = len(self.critic_model_group._actor_handlers)
                    effective_actors = num_actors // self.critic_model_group.duplicate_actors
                    chunk_size = len(experiences) // (effective_actors * self.args.micro_train_batch_size)

                    sample_size = chunk_size * effective_actors * self.args.micro_train_batch_size
                    sample_chunk_size = sample_size // self.args.n_samples_per_prompt
                    actual_sample_size = sample_chunk_size * self.args.n_samples_per_prompt
                    critic_experiences = experiences[:actual_sample_size]

                    logger.info({
                        'INFO': '##DYNAMIC_BATCH_SIZE_CRITIC##',
                        'VALUE': len(critic_experiences)
                    })
                else:
                    critic_experiences = experiences

                if self.args.use_seq_balancing:
                    seqlen = [exp.attention_mask.sum().tolist() for exp in critic_experiences]
                    num_actors = len(self.critic_model_group._actor_handlers)
                    k_partitions = num_actors // self.critic_model_group.duplicate_actors
                    critic_experience_ids = get_seqlen_balanced_partitions(seqlen, k_partitions, equal_size=True)
                    experiences_critic = []
                    for group_ids in critic_experience_ids:
                        for idx in group_ids:
                            experiences_critic.append(critic_experiences[idx])
                else:
                    experiences_critic = critic_experiences

                # balance experiences across dp
                if args.use_dynamic_batch:
                    experiences_critic = balance_experiences(experiences_critic, args, mode='critic')

                refs.extend(
                    self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences_critic)
                )
            ray.get(refs)

            if self.args.use_global_token_level_loss:
                if self.args.use_loss_mask:
                    global_response_length = sum([(exp.action_mask*exp.loss_mask.unsqueeze(dim=-1)).sum().tolist() for exp in experiences])
                else:
                    global_response_length = sum([exp.action_mask.sum().tolist() for exp in experiences])
            else:
                global_response_length = None

            if self.args.use_exp_before_queue:
                # Wait for training to be allowed
                ray.get(self.signal_actor.wait_train.remote())

            status = self.ppo_train(steps, global_response_length=global_response_length)

            if "kl" in status:
                self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

            if global_response_length is not None:
                status['global_response_length'] = global_response_length
            status['generate_eposide'] = queue_info['dump_idx']
            
            # Add generated samples to status dictionary
            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = queue_info['pass_rate']
            logger.info(f"✨ Global step {steps}: {status}")
            status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]
            if self.args.use_length_acc_balance:
                status['filtered_samples_nums'] = queue_info['filtered_samples_nums']

            # logs/checkpoints
            client_states = {
                "global_step": steps,
                "episode": episode,
                "data_loader_state_dict": data_loader_state_dict,
            }
            self.save_logs_and_checkpoints(args, steps, None, status, client_states)

            steps = steps + 1
            is_train = True

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()


@ray.remote
class PPOTrainerAsync:
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: PPORayActorGroup,
        critic_model_group: PPORayActorGroup,
        reward_model_group: PPORayActorGroup,
        reference_model_group: PPORayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.args = strategy.args
        self.strategy = strategy
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.vllm_engines = vllm_engines
        self.prompt_max_len = prompt_max_len

        # Create signal actor for synchronization
        self.signal_actor = SignalActor.remote()

        # if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
        #     from openrlhf.utils.remote_rm_utils import RemoteRewardModel

        #     self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)
        # else:
        #     self.remote_reward_model = None
        self.remote_reward_model = None

        self.generator_actor = GenerateSamplesActor.remote(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            signal_actor=self.signal_actor,
            **generate_kwargs,
        )

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt

        self.trainer_actor = TrainingActor.remote(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            signal_actor=self.signal_actor,
            remote_reward_model=self.remote_reward_model,
            **generate_kwargs,
        )

        from ray.util.queue import Queue

        # the max size is used to control the degree of off-policy
        async_queue_size = int(os.environ.get("OPENRLHF_ASYNC_QUEUE_SIZE", 1))
        self.queue = Queue(maxsize=async_queue_size)
        logger.info({
            'INFO': '##SIZE-OF-QUEUE-SIZE',
            'VALUE': async_queue_size
        })

        self.train_meta_queue = Queue(maxsize=async_queue_size)

    def fit(self) -> None:
        args = self.args

        # Update initial weights to vLLM engines
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            ray.get(self.trainer_actor._broadcast_to_vllm.remote())
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]

        # Launch async training
        remote_reward_model = None
        generator_actor_ref = self.generator_actor.fit.remote(
            episode, self.queue, data_loader_state_dict, self.train_meta_queue
        )
        trainer_actor_ref = self.trainer_actor.fit.remote(self.queue, steps, self.train_meta_queue)
        ray.get([generator_actor_ref, trainer_actor_ref])

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
