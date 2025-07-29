import asyncio, uuid
import os, json, time
from typing import Optional, Any, List, Dict, Tuple
from typing import List, Dict, Union, Any
import ray

from openrlhf.utils import get_tokenizer
from .vllm_engine import BaseLLMRayActor
from openrlhf.async_pipline.process_request import GenerateRequest, process_batch_requests
from openrlhf.utils.logging_utils import init_logger
from openrlhf.async_pipline.show_timer import Timer

logger = init_logger(__name__)

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 32))  # 最大并发数，可根据需要调整
MAX_VLLM_BATCHSIZE = int(os.getenv("MAX_VLLM_BATCHSIZE", 32))  # 最大并发数，可根据需要调整
logger.info({
    'INFO': "##MAX_CONCURRENT##",
    'VALUE': MAX_CONCURRENT,
    'BATCHSIZE': MAX_VLLM_BATCHSIZE
})


@ray.remote
class AgentInstance:
    def __init__(self, agent_func_path):
        if agent_func_path.endswith(".py"):
            import importlib.util

            spec = importlib.util.spec_from_file_location("step", agent_func_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            self.agent_step = agent_module.step
        else:
            raise ValueError("Agent path must be a Python file")

    async def step(self, state, action, label):
        return await self.agent_step(state, action, label)


@ray.remote
def get_tokenize_text_len(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0])


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        self.agent_func_path = kwargs.pop("agent_func_path")

        # Initialize super class
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()
        self.batch_size = MAX_VLLM_BATCHSIZE

        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)  # 实例级共享

        os.environ["VLLM_USE_V1"] = "1"
        import vllm

        assert vllm.__version__ > "0.8.5", "Asyn VLLM version must be greater than 0.8.5"

        self.model_path = self.kwargs['model']

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def generate_async_server(self, request: GenerateRequest, sampling_params, request_id):
        # Send the request to the LLM engine.
        from vllm.inputs import TokensPrompt
        async with self.semaphore:  # 使用共享信号量
        # async with asyncio.Semaphore(MAX_CONCURRENT):  # 实例级共享
            stream = self.llm.generate(
                request_id=str(request_id),
                prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
                sampling_params=sampling_params,
            )

            # Consume the stream until the request is finished.
            # 移入循环内部确保作用域隔离
            final_output = None
            async for request_output in stream:
                final_output = request_output
            if final_output is None:
                raise RuntimeError(f"Empty stream for request_id: {request_id}")
            
            assert final_output.request_id == request_id
            output = [{
                'outputs':[
                    {
                        "text": final_output.outputs[0].text,
                        "token_ids": final_output.outputs[0].token_ids,
                        "stop_reason": final_output.outputs[0].stop_reason,
                        "finish_reason": final_output.outputs[0].finish_reason,
                    }
                ],
                "prompt_token_ids": final_output.prompt_token_ids,
                "request_id": final_output.request_id
            }]
            return output

    async def async_llm_generate(self, request: GenerateRequest):
        # 实际生成逻辑
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            include_stop_str_in_output=request.include_stop_str_in_output,
            stop=request.stop,
            skip_special_tokens=False,
            allowed_token_ids=request.allowed_token_ids if request.allowed_token_ids is not None else None
        )

        # request_id = str(uuid.uuid4())+request.uuids
        request_id = f"{time.time_ns()}-{uuid.uuid4()}"
        response = await self.generate_async_server(request, sampling_params, request_id)
        return response

    def build_requests(self, prompts, prompt_ids, sampling_params, labels=None, requests_ranks=None, max_length=None, tokenizer=None):
        request_list = []
        for idx, (prompt, prompt_id) in enumerate(zip(prompts, prompt_ids)):
            if labels is not None:
                if labels[idx] is not None:
                    label_dict = json.loads(labels[idx])
                    uuid_str = label_dict.get('uuid', str(uuid.uuid4()))
                    env_func = label_dict.get('env_func', 'math_tir_async')
                else:
                    env_func = 'math_tir_async'
                    uuid_str = str(uuid.uuid4())
                    label_dict = {
                        'uuid': uuid_str,
                        'env_func': env_func
                    }
            else:
                env_func = 'math_tir_async'
                uuid_str = str(uuid.uuid4())
                label_dict = {
                    'uuid': uuid_str,
                    'env_func': env_func
                }
            
            if requests_ranks is not None:
                request_rank = requests_ranks[idx]
            else:
                request_rank = 0
            if 'Qwen3' in self.model_path and tokenizer is not None:
                allowed_token_ids = list(range(tokenizer.vocab_size)) + list(tokenizer.added_tokens_decoder.keys())
            else:
                allowed_token_ids = None
            request = GenerateRequest(
                prompts=[prompt],
                prompt_token_ids=prompt_id,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                stop=sampling_params.stop,
                uuids=uuid_str+f'####idx:{idx}',
                env_func=env_func,
                label=json.dumps(label_dict, ensure_ascii=False),
                request_rank=request_rank,
                max_length=max_length,
                allowed_token_ids=allowed_token_ids
            )
            request_list.append(request)
        return request_list

    def group_requests(self, data_list: List[Dict]):
        requests_dict = {}
        for data in data_list:
            env_func = data.env_func
            if env_func not in requests_dict:
                requests_dict[env_func] = []
            requests_dict[env_func].append(data)
        logger.info({
                'INFO': f'##SIZE-OF-REQUESTS-DICT:{len(requests_dict)}##',
        })
        return requests_dict

    def _create_batches(self, data_list: Union[List[Dict[Any, Any]], Dict[Any, List[Any]]]) -> List[Tuple[int, List[Dict]]]:
        """将数据分成 batch，返回 [(start_idx, batch), ...]"""
        batches = []
        if isinstance(data_list, list):
            for i in range(0, len(data_list), self.batch_size):
                batch = data_list[i:i + self.batch_size]
                batches.append((i, batch))
            if i + self.batch_size < len(data_list) - 1:
                batches.append((i+1, data_list[i + self.batch_size:]))
        elif isinstance(data_list, dict):
            for env_func in data_list:
                for i in range(0, len(data_list[env_func]), self.batch_size):
                    batch = data_list[env_func][i:i + self.batch_size]
                    batches.append((i, batch))
                if i + self.batch_size < len(data_list[env_func]) - 1:
                    batches.append((i+1, data_list[env_func][i + self.batch_size:]))
        else:
            raise ValueError("data_list must be a list or dict")
        return batches

    async def add_requests(self, sampling_params, prompts, prompt_ids, labels, max_length, hf_tokenizer=None, max_steps=10000):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the step function.
        Results are streamed back as each agent completes its execution.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of prompts to process
            labels: List of labels corresponding to prompts
            max_steps: Maximum number of interaction steps
            micro_forward_batch_size: Number of prompts to process in each concurrent task
        """
        all_requests = self.build_requests(prompts=prompts, prompt_ids=prompt_ids, 
                                            sampling_params=sampling_params, 
                                            labels=labels,
                                            requests_ranks=None,
                                            max_length=max_length,
                                            tokenizer=hf_tokenizer)
        if labels is not None:
            all_requests = self.group_requests(all_requests)
        batches = self._create_batches(all_requests)
        response_tasks = []
        for start_idx, batch in batches:
            env_func = batch[0].env_func
            response_tasks.append(process_batch_requests(self.async_llm_generate, start_idx, batch, env_func=env_func, tokenizer=hf_tokenizer))

        async with Timer("##ASYNC-ROLLOUT-WHOLE-ROLLOUT##"):
            results_raw = await asyncio.gather(*response_tasks)
        flat_results = []
        for result_raw in results_raw:
            successful_results, failed_results = result_raw
            for item in successful_results:
                flat_results.append(item)
        responses = [result[1][1] for result in flat_results]
        responses.sort(key=lambda x: int(x.request_id.split('####idx:')[-1]))

        queue_tasks = []
        for response in responses:
            queue_tasks.append(self.result_queue.put(response))
        async with Timer("##ASYNC-ROLLOUT-ENTER-QUEUE##"):
            await asyncio.gather(*queue_tasks)

        logger.info({
            'INFO': f'##END-TO-ROLLOUT##',
        })
        

    async def get_responses(self):
        """
        Synchronously get all completed agent results from the queue.
        Waits for all tasks to complete before returning results.
        Returns: List of all completed agent results.
        """
        # Get all results from the queue
        results = []
        while not self.result_queue.empty():
            try:
                results.append(await self.result_queue.get())
            except asyncio.QueueEmpty:
                break
        return results
