import ray
import zmq
import asyncio
import numpy as np
from zmq.asyncio import Context

# 初始化 Ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

@ray.remote
class Publisher:
    def __init__(self, port=5557):
        self.ctx = Context.instance()
        self.socket = self.ctx.socket(zmq.PUB)
        self.ip_address = ray._private.services.get_node_ip_address()
        self.socket.bind(f"tcp://{self.ip_address}:{port}")

    def _node_ip_address(self):
        return self.ip_address  # 直接返回已存储的IP

    async def publish_data(self, data):
        await self.socket.send_pyobj(data)
        print(f"Published data: {data[:5]}...")

@ray.remote
class Worker:
    def __init__(self, pub_host, port=5557):
        self.ctx = Context.instance()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect(f"tcp://{pub_host}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    async def process(self):
        data = await self.socket.recv_pyobj()  # 移除了无限循环
        result = np.mean(data)
        print(f"Worker processed data: {result:.4f}")
        return result

@ray.remote
class Aggregator:
    def __init__(self, port=5558):
        self.ctx = Context.instance()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")

    async def publish_result(self, result):
        await self.socket.send_pyobj(result)
        print(f"Aggregated result: {result:.4f}")

async def main():
    publisher = Publisher.remote()
    aggregator = Aggregator.remote()
    
    pub_host = ray.get(publisher._node_ip_address.remote())
    workers = [Worker.remote(pub_host) for _ in range(10)]
    
    for i in range(5):
        # 先启动Worker的process任务
        worker_tasks = [worker.process.remote() for worker in workers]
        
        # 给予足够时间让Worker连接
        await asyncio.sleep(0.2)
        
        # 生成并发布数据
        data = np.random.rand(1000)
        await publisher.publish_data.remote(data)
        
        # 收集结果
        results = await asyncio.gather(*worker_tasks)
        
        final_result = sum(results) / len(results)
        await aggregator.publish_result.remote(final_result)
        print(f"Round {i+1}: Final {final_result:.4f}")

if __name__ == "__main__":
    asyncio.run(main())