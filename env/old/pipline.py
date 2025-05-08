import queue
import threading

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_0304_vllm083'))
from env.vllm_serve import GenerateRequest
from typing import Any, List
from env.vllm_client import VLLMClient
import time

class VLLMPIPEServer(object):
    def __init__(self, client):
        self.client = client
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.producer_done = threading.Event()
        self.consumer_threads = []

    def _produce_tasks(self, request_list):
        """同步生成任务到队列"""
        for request in request_list:
            self.request_queue.put(request)
        self.producer_done.set()

    def _process_task(self, request: GenerateRequest):
        """处理单个提示词并返回结果"""
        for _ in range(10):
            try:
                response = self.client.generate(request)
                result = response.choices[0].message.content
                return result
            except Exception as e:
                time.sleep(1.0)
                print(f"Request failed: {e}")
                return None

    def _consumer_loop(self):
        """消费者线程的主循环"""
        while True:
            # 检查生产者是否完成且队列已空
            if self.producer_done.is_set() and self.request_queue.empty():
                break
            
            try:
                request = self.request_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            try:
                result = self._process_task(request)
                self.result_queue.put(result)
            finally:
                self.request_queue.task_done()

    def process_requests(self, request_list: List[Any], num_requests: int, num_consumers: int):
        """
        处理请求的主入口
        :param num_requests: 总请求数量
        :param num_consumers: 消费者线程数量
        """
        # 同步生成任务（无生产者线程开销）
        self._produce_tasks(request_list)
        
        # 启动消费者线程
        self.consumer_threads = [
            threading.Thread(target=self._consumer_loop)
            for _ in range(num_consumers)
        ]
        for thread in self.consumer_threads:
            thread.start()
        
        # 等待所有任务处理完成
        self.request_queue.join()
        
        # 确保所有消费者线程退出
        for thread in self.consumer_threads:
            thread.join()

    def get_results(self):
        """获取处理结果列表"""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results