# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import heapq
from typing import List, Tuple

import torch
from torch import distributed as dist


def karmarkar_karp(
    seqlen_list: List[int], 
    k_partitions: int, 
    equal_size: bool, 
    max_seqlen_per_partition: int = None
):
    """
    Karmarkar-Karp算法实现，支持分区最大长度限制
    """
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True, key=lambda x: x.sum)

        def get_partitions(self):
            return [[idx for idx, _ in s.items] for s in self.sets]

        def can_merge(self, other, max_limit: int) -> bool:
            """检查合并是否会导致任何分区超过最大长度"""
            for i in range(self.k):
                merged_sum = self.sets[i].sum + other.sets[self.k-1-i].sum
                if merged_sum > max_limit:
                    return False
            return True

        def merge_safely(self, other):
            """安全合并（假设已通过can_merge检查）"""
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k-1-i])
            self.sets = sorted(self.sets, reverse=True, key=lambda x: x.sum)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum if self.sets else 0

        def __lt__(self, other):
            if self.spread != other.spread:
                return self.spread > other.spread  # 大根堆，优先处理差异大的状态
            return self.sets[0].sum > other.sets[0].sum  # 相同差异时优先处理最大集合大的状态

    # 预处理：检查单个元素是否超过限制
    if max_seqlen_per_partition is not None:
        for seqlen in seqlen_list:
            if seqlen > max_seqlen_per_partition:
                raise ValueError(f"Single element {seqlen} exceeds max_seqlen_per_partition {max_seqlen_per_partition}")

    sorted_seqlen = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)], reverse=True)
    states_pq = []

    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, "Equal size requires divisible length"
        chunk_size = len(seqlen_list) // k_partitions
        for i in range(k_partitions):
            start = i * chunk_size
            end = (i+1) * chunk_size
            items = [(idx, seqlen) for seqlen, idx in sorted_seqlen[start:end]]
            states_pq.append(State(items, k_partitions))
    else:
        for seqlen, idx in sorted_seqlen:
            states_pq.append(State([(idx, seqlen)], k_partitions))

    # 转换为优先队列
    heapq.heapify(states_pq)

    while len(states_pq) > 1:
        state1 = heapq.heappop(states_pq)
        state2 = heapq.heappop(states_pq)
        
        # 检查合并是否安全
        if max_seqlen_per_partition is not None and not state1.can_merge(state2, max_seqlen_per_partition):
            # 尝试不同的合并顺序（简单重试机制，复杂场景需更智能策略）
            retry_states = [state1, state2] + list(states_pq)
            heapq.heapify(retry_states)
            states_pq = retry_states
            continue  # 跳过本次合并，尝试下一个最小差异的状态
        
        # 执行安全合并
        state1.merge_safely(state2)
        heapq.heappush(states_pq, state1)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()

    # 最终验证（处理可能的浮点误差或逻辑漏洞）
    if max_seqlen_per_partition is not None:
        for part in partitions:
            total = sum(seqlen_list[idx] for idx in part)
            if total > max_seqlen_per_partition + 1e-6:  # 允许极小浮点误差
                raise RuntimeError(f"Partition exceeds max length: {total} > {max_seqlen_per_partition}")

    return partitions


def greedy_partition(
    seqlen_list: List[int], 
    k_partitions: int, 
    equal_size: bool, 
    max_seqlen_per_partition: int = None
):
    """
    贪心分区算法，支持最大长度限制
    """
    if max_seqlen_per_partition is not None:
        for seqlen in seqlen_list:
            if seqlen > max_seqlen_per_partition:
                raise ValueError(f"Single element {seqlen} exceeds max_seqlen_per_partition {max_seqlen_per_partition}")

    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_items = sorted([(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)], reverse=True)
    
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0] * k_partitions

    for seqlen, idx in sorted_items:
        original_seqlen = seqlen - bias if equal_size else seqlen
        best_idx = None
        
        # 寻找可以容纳的最小分区
        for i in range(k_partitions):
            if partition_sums[i] + original_seqlen <= (max_seqlen_per_partition or float('inf')):
                if best_idx is None or partition_sums[i] < partition_sums[best_idx]:
                    best_idx = i
        
        if best_idx is None:
            raise RuntimeError("No valid partition found that satisfies max length constraint")
        
        partitions[best_idx].append(idx)
        partition_sums[best_idx] += original_seqlen

    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, "Equal size requires divisible length"
        for part in partitions:
            assert len(part) == len(seqlen_list) // k_partitions, "Unequal partition size"

    return partitions


def get_seqlen_balanced_partitions(
    seqlen_list: List[int], 
    k_partitions: int, 
    equal_size: bool, 
    max_seqlen_per_partition: int = None
):
    """
    计算序列长度平衡的分区，支持最大分区长度限制
    
    Args:
        seqlen_list: 序列长度列表
        k_partitions: 分区数量
        equal_size: 是否要求每个分区元素数量相同
        max_seqlen_per_partition: 每个分区的最大总长度限制（可选）
    
    Returns:
        分区结果列表，每个子列表包含原始索引（已排序）
    
    Raises:
        ValueError: 当单个元素超过最大长度或无法形成有效分区时
        AssertionError: 基本参数校验失败时
    """
    assert len(seqlen_list) >= k_partitions, f"Items {len(seqlen_list)} < Partitions {k_partitions}"
    
    if max_seqlen_per_partition is not None and max_seqlen_per_partition < min(seqlen_list):
        raise ValueError("Max length must be at least the smallest element")

    # 选择分区算法（Karmarkar-Karp更适合平衡总和，贪心更适合快速分配）
    if equal_size or len(seqlen_list) <= 2 * k_partitions:
        partitions = greedy_partition(
            seqlen_list, k_partitions, equal_size, max_seqlen_per_partition
        )
    else:
        partitions = karmarkar_karp(
            seqlen_list, k_partitions, equal_size, max_seqlen_per_partition
        )

    # 后处理：排序索引并验证完整性
    sorted_partitions = []
    seen = set()
    for part in partitions:
        sorted_part = sorted(part)
        total = sum(seqlen_list[idx] for idx in sorted_part)
        if max_seqlen_per_partition is not None and total > max_seqlen_per_partition:
            raise AssertionError(f"Partition sum {total} exceeds limit {max_seqlen_per_partition}")
        sorted_partitions.append(sorted_part)
        seen.update(sorted_part)
    
    assert len(seen) == len(seqlen_list), "Missing indices in partitions"
    assert len(sorted_partitions) == k_partitions, "Incorrect number of partitions"
    
    return sorted_partitions