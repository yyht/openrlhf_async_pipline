import ray
import torch

# 初始化Ray
ray.init()

# 定义一个返回列表的函数
def return_list():
    return [torch.tensor([1,2,3]), torch.tensor([1,2,3])]

# 创建多个任务
object_refs = [ray.put(return_list()) for _ in range(3)]

# 获取结果
ray.get(object_refs)

import time
time.sleep(1.0)

results = ray.get(object_refs)

# 打印结果
print("所有任务的返回结果:", results)

# 关闭Ray
ray.shutdown()
