
import ray
import asyncio
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from ray.util.queue import Queue
from dataclasses import dataclass
from enum import Enum

# 初始化Ray
ray.init(ignore_reinit_error=True)

class CommandType(Enum):
    GET = "GET"
    SET = "SET"
    DEL = "DEL"
    LPUSH = "LPUSH"
    RPUSH = "RPUSH"
    LPOP = "LPOP"
    RPOP = "RPOP"
    LLEN = "LLEN"
    LRANGE = "LRANGE"
    HSET = "HSET"
    HGET = "HGET"
    HDEL = "HDEL"
    HGETALL = "HGETALL"
    SADD = "SADD"
    SREM = "SREM"
    SMEMBERS = "SMEMBERS"
    EXPIRE = "EXPIRE"
    TTL = "TTL"
    PING = "PING"

@dataclass
class RedisCommand:
    cmd_type: CommandType
    key: str
    args: List[Any]
    client_id: str
    timestamp: float

@dataclass
class RedisResponse:
    success: bool
    data: Any = None
    error: str = None
    client_id: str = None

@ray.remote
class RedisNode:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.data: Dict[str, Any] = {}
        self.expires: Dict[str, float] = {}
        self.command_queue = Queue(maxsize=1000)
        self.response_queues: Dict[str, Queue] = {}
        self.running = True
        
    async def start(self):
        """启动节点处理循环"""
        while self.running:
            try:
                # 非阻塞获取命令
                command = await self._get_command_async()
                if command:
                    response = await self._execute_command(command)
                    await self._send_response(command.client_id, response)
                else:
                    await asyncio.sleep(0.001)  # 短暂休眠避免CPU占用过高
            except Exception as e:
                print(f"Node {self.node_id} error: {e}")
                await asyncio.sleep(0.1)
    
    async def _get_command_async(self) -> Optional[RedisCommand]:
        """异步获取命令"""
        try:
            # 使用非阻塞方式获取命令
            if not self.command_queue.empty():
                return self.command_queue.get_nowait()
        except:
            pass
        return None
    
    async def _send_response(self, client_id: str, response: RedisResponse):
        """发送响应给客户端"""
        response.client_id = client_id
        if client_id in self.response_queues:
            try:
                self.response_queues[client_id].put_nowait(response)
            except:
                # 队列满了，丢弃旧的响应
                try:
                    self.response_queues[client_id].get_nowait()
                    self.response_queues[client_id].put_nowait(response)
                except:
                    pass
    
    def add_client(self, client_id: str):
        """添加客户端响应队列"""
        if client_id not in self.response_queues:
            self.response_queues[client_id] = Queue(maxsize=100)
    
    def remove_client(self, client_id: str):
        """移除客户端响应队列"""
        if client_id in self.response_queues:
            del self.response_queues[client_id]
    
    def submit_command(self, command: RedisCommand):
        """提交命令到队列"""
        try:
            self.command_queue.put_nowait(command)
            return True
        except:
            return False
    
    def get_response(self, client_id: str) -> Optional[RedisResponse]:
        """获取客户端响应"""
        if client_id in self.response_queues:
            try:
                return self.response_queues[client_id].get_nowait()
            except:
                pass
        return None
    
    async def _execute_command(self, command: RedisCommand) -> RedisResponse:
        """执行Redis命令"""
        try:
            # 清理过期键
            self._cleanup_expired_keys()
            
            if command.cmd_type == CommandType.PING:
                return RedisResponse(success=True, data="PONG")
            
            elif command.cmd_type == CommandType.GET:
                if self._is_expired(command.key):
                    return RedisResponse(success=True, data=None)
                return RedisResponse(success=True, data=self.data.get(command.key))
            
            elif command.cmd_type == CommandType.SET:
                value = command.args[0]
                self.data[command.key] = value
                # 处理过期时间
                if len(command.args) > 1:
                    ttl = command.args[1]
                    self.expires[command.key] = time.time() + ttl
                return RedisResponse(success=True, data="OK")
            
            elif command.cmd_type == CommandType.DEL:
                deleted = 0
                if command.key in self.data:
                    del self.data[command.key]
                    deleted += 1
                if command.key in self.expires:
                    del self.expires[command.key]
                return RedisResponse(success=True, data=deleted)
            
            elif command.cmd_type == CommandType.LPUSH:
                if command.key not in self.data:
                    self.data[command.key] = []
                if not isinstance(self.data[command.key], list):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                for value in command.args:
                    self.data[command.key].insert(0, value)
                return RedisResponse(success=True, data=len(self.data[command.key]))
            
            elif command.cmd_type == CommandType.RPUSH:
                if command.key not in self.data:
                    self.data[command.key] = []
                if not isinstance(self.data[command.key], list):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                for value in command.args:
                    self.data[command.key].append(value)
                return RedisResponse(success=True, data=len(self.data[command.key]))
            
            elif command.cmd_type == CommandType.LPOP:
                if command.key not in self.data or not self.data[command.key]:
                    return RedisResponse(success=True, data=None)
                if not isinstance(self.data[command.key], list):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                return RedisResponse(success=True, data=self.data[command.key].pop(0))
            
            elif command.cmd_type == CommandType.RPOP:
                if command.key not in self.data or not self.data[command.key]:
                    return RedisResponse(success=True, data=None)
                if not isinstance(self.data[command.key], list):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                return RedisResponse(success=True, data=self.data[command.key].pop())
            
            elif command.cmd_type == CommandType.LLEN:
                if command.key not in self.data:
                    return RedisResponse(success=True, data=0)
                if not isinstance(self.data[command.key], list):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                return RedisResponse(success=True, data=len(self.data[command.key]))
            
            elif command.cmd_type == CommandType.LRANGE:
                if command.key not in self.data:
                    return RedisResponse(success=True, data=[])
                if not isinstance(self.data[command.key], list):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                start, end = command.args[0], command.args[1]
                return RedisResponse(success=True, data=self.data[command.key][start:end+1])
            
            elif command.cmd_type == CommandType.HSET:
                if command.key not in self.data:
                    self.data[command.key] = {}
                if not isinstance(self.data[command.key], dict):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                field, value = command.args[0], command.args[1]
                self.data[command.key][field] = value
                return RedisResponse(success=True, data=1)
            
            elif command.cmd_type == CommandType.HGET:
                if command.key not in self.data:
                    return RedisResponse(success=True, data=None)
                if not isinstance(self.data[command.key], dict):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                field = command.args[0]
                return RedisResponse(success=True, data=self.data[command.key].get(field))
            
            elif command.cmd_type == CommandType.HGETALL:
                if command.key not in self.data:
                    return RedisResponse(success=True, data={})
                if not isinstance(self.data[command.key], dict):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                return RedisResponse(success=True, data=self.data[command.key])
            
            elif command.cmd_type == CommandType.SADD:
                if command.key not in self.data:
                    self.data[command.key] = set()
                if not isinstance(self.data[command.key], set):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                added = 0
                for value in command.args:
                    if value not in self.data[command.key]:
                        self.data[command.key].add(value)
                        added += 1
                return RedisResponse(success=True, data=added)
            
            elif command.cmd_type == CommandType.SREM:
                if command.key not in self.data:
                    return RedisResponse(success=True, data=0)
                if not isinstance(self.data[command.key], set):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                removed = 0
                for value in command.args:
                    if value in self.data[command.key]:
                        self.data[command.key].remove(value)
                        removed += 1
                return RedisResponse(success=True, data=removed)
            
            elif command.cmd_type == CommandType.SMEMBERS:
                if command.key not in self.data:
                    return RedisResponse(success=True, data=set())
                if not isinstance(self.data[command.key], set):
                    return RedisResponse(success=False, error="WRONGTYPE Operation against a key holding the wrong kind of value")
                return RedisResponse(success=True, data=list(self.data[command.key]))
            
            elif command.cmd_type == CommandType.EXPIRE:
                if command.key not in self.data:
                    return RedisResponse(success=True, data=0)
                ttl = command.args[0]
                self.expires[command.key] = time.time() + ttl
                return RedisResponse(success=True, data=1)
            
            elif command.cmd_type == CommandType.TTL:
                if command.key not in self.data:
                    return RedisResponse(success=True, data=-2)
                if command.key not in self.expires:
                    return RedisResponse(success=True, data=-1)
                ttl = self.expires[command.key] - time.time()
                return RedisResponse(success=True, data=int(ttl) if ttl > 0 else -2)
            
            else:
                return RedisResponse(success=False, error=f"Unknown command: {command.cmd_type}")
                
        except Exception as e:
            return RedisResponse(success=False, error=str(e))
    
    def _is_expired(self, key: str) -> bool:
        """检查键是否过期"""
        if key in self.expires:
            if time.time() > self.expires[key]:
                if key in self.data:
                    del self.data[key]
                del self.expires[key]
                return True
        return False
    
    def _cleanup_expired_keys(self):
        """清理过期键"""
        current_time = time.time()
        expired_keys = [key for key, expire_time in self.expires.items() if current_time > expire_time]
        for key in expired_keys:
            if key in self.data:
                del self.data[key]
            del self.expires[key]
    
    def stop(self):
        """停止节点"""
        self.running = False
    
    def get_stats(self):
        """获取节点统计信息"""
        return {
            "node_id": self.node_id,
            "keys_count": len(self.data),
            "expires_count": len(self.expires),
            "queue_size": self.command_queue.qsize() if hasattr(self.command_queue, 'qsize') else 0
        }

@ray.remote
class RedisCluster:
    def __init__(self, num_nodes: int = 3):
        self.num_nodes = num_nodes
        self.nodes = []
        self.node_tasks = []
        self.hash_ring = {}
        self._init_cluster()
    
    def _init_cluster(self):
        """初始化集群"""
        # 创建节点
        for i in range(self.num_nodes):
            node = RedisNode.remote(f"node_{i}")
            self.nodes.append(node)
            # 启动节点处理循环
            task = node.start.remote()
            self.node_tasks.append(task)
        
        # 构建哈希环
        self._build_hash_ring()
    
    def _build_hash_ring(self):
        """构建一致性哈希环"""
        self.hash_ring = {}
        for i, node in enumerate(self.nodes):
            # 为每个节点创建多个虚拟节点
            for j in range(100):
                key = f"node_{i}_{j}"
                hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                self.hash_ring[hash_value] = i
    
    def _get_node_for_key(self, key: str) -> int:
        """根据键获取对应的节点索引"""
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # 找到大于等于key_hash的最小节点
        sorted_hashes = sorted(self.hash_ring.keys())
        for hash_value in sorted_hashes:
            if hash_value >= key_hash:
                return self.hash_ring[hash_value]
        
        # 如果没找到，返回第一个节点（环形）
        return self.hash_ring[sorted_hashes[0]]
    
    def add_client(self, client_id: str):
        """在所有节点上添加客户端"""
        for node in self.nodes:
            node.add_client.remote(client_id)
    
    def remove_client(self, client_id: str):
        """从所有节点移除客户端"""
        for node in self.nodes:
            node.remove_client.remote(client_id)
    
    def execute_command(self, command: RedisCommand) -> bool:
        """执行命令"""
        node_index = self._get_node_for_key(command.key)
        node = self.nodes[node_index]
        return ray.get(node.submit_command.remote(command))
    
    def get_response(self, client_id: str) -> Optional[RedisResponse]:
        """获取响应"""
        # 从所有节点尝试获取响应
        for node in self.nodes:
            response = ray.get(node.get_response.remote(client_id))
            if response:
                return response
        return None
    
    def get_cluster_stats(self):
        """获取集群统计信息"""
        stats = []
        for node in self.nodes:
            node_stats = ray.get(node.get_stats.remote())
            stats.append(node_stats)
        return stats
    
    def shutdown(self):
        """关闭集群"""
        for node in self.nodes:
            node.stop.remote()

class RayRedisClient:
    def __init__(self, cluster_ref):
        self.cluster = cluster_ref
        self.client_id = f"client_{int(time.time() * 1000000)}"
        ray.get(self.cluster.add_client.remote(self.client_id))
    
    async def _execute_command(self, cmd_type: CommandType, key: str, *args) -> Any:
        """执行命令"""
        command = RedisCommand(
            cmd_type=cmd_type,
            key=key,
            args=list(args),
            client_id=self.client_id,
            timestamp=time.time()
        )
        
        # 提交命令
        success = ray.get(self.cluster.execute_command.remote(command))
        if not success:
            raise Exception("Failed to submit command")
        
        # 等待响应
        max_retries = 100
        for _ in range(max_retries):
            response = ray.get(self.cluster.get_response.remote(self.client_id))
            if response:
                if response.success:
                    return response.data
                else:
                    raise Exception(response.error)
            await asyncio.sleep(0.01)
        
        raise Exception("Command timeout")
    
    # 字符串操作
    async def get(self, key: str) -> Optional[str]:
        return await self._execute_command(CommandType.GET, key)
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> str:
        if ttl:
            return await self._execute_command(CommandType.SET, key, value, ttl)
        return await self._execute_command(CommandType.SET, key, value)
    
    async def delete(self, key: str) -> int:
        return await self._execute_command(CommandType.DEL, key)
    
    # 列表操作
    async def lpush(self, key: str, *values) -> int:
        return await self._execute_command(CommandType.LPUSH, key, *values)
    
    async def rpush(self, key: str, *values) -> int:
        return await self._execute_command(CommandType.RPUSH, key, *values)
    
    async def lpop(self, key: str) -> Optional[str]:
        return await self._execute_command(CommandType.LPOP, key)
    
    async def rpop(self, key: str) -> Optional[str]:
        return await self._execute_command(CommandType.RPOP, key)
    
    async def llen(self, key: str) -> int:
        return await self._execute_command(CommandType.LLEN, key)
    
    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        return await self._execute_command(CommandType.LRANGE, key, start, end)
    
    # 哈希操作
    async def hset(self, key: str, field: str, value: str) -> int:
        return await self._execute_command(CommandType.HSET, key, field, value)
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        return await self._execute_command(CommandType.HGET, key, field)
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        return await self._execute_command(CommandType.HGETALL, key)
    
    # 集合操作
    async def sadd(self, key: str, *values) -> int:
        return await self._execute_command(CommandType.SADD, key, *values)
    
    async def srem(self, key: str, *values) -> int:
        return await self._execute_command(CommandType.SREM, key, *values)
    
    async def smembers(self, key: str) -> List[str]:
        return await self._execute_command(CommandType.SMEMBERS, key)
    
    # 过期时间
    async def expire(self, key: str, ttl: int) -> int:
        return await self._execute_command(CommandType.EXPIRE, key, ttl)
    
    async def ttl(self, key: str) -> int:
        return await self._execute_command(CommandType.TTL, key)
    
    # 连接测试
    async def ping(self) -> str:
        return await self._execute_command(CommandType.PING, "")
    
    def close(self):
        """关闭客户端"""
        ray.get(self.cluster.remove_client.remote(self.client_id))

# 使用示例
async def main():
    # 创建Redis集群
    cluster = RedisCluster.remote(num_nodes=3)
    
    # 创建客户端
    client = RayRedisClient(cluster)
    
    try:
        # 测试基本操作
        print("=== 基本字符串操作 ===")
        await client.set("key1", "value1")
        result = await client.get("key1")
        print(f"GET key1: {result}")
        
        # 测试带TTL的设置
        await client.set("temp_key", "temp_value", ttl=5)
        print(f"TTL temp_key: {await client.ttl('temp_key')}")
        
        print("\n=== 列表操作 ===")
        await client.lpush("list1", "item1", "item2", "item3")
        print(f"LLEN list1: {await client.llen('list1')}")
        print(f"LRANGE list1 0 -1: {await client.lrange('list1', 0, -1)}")
        print(f"LPOP list1: {await client.lpop('list1')}")
        
        print("\n=== 哈希操作 ===")
        await client.hset("hash1", "field1", "value1")
        await client.hset("hash1", "field2", "value2")
        print(f"HGET hash1 field1: {await client.hget('hash1', 'field1')}")
        print(f"HGETALL hash1: {await client.hgetall('hash1')}")
        
        print("\n=== 集合操作 ===")
        await client.sadd("set1", "member1", "member2", "member3")
        print(f"SMEMBERS set1: {await client.smembers('set1')}")
        await client.srem("set1", "member2")
        print(f"SMEMBERS set1 after SREM: {await client.smembers('set1')}")
        
        print("\n=== 连接测试 ===")
        print(f"PING: {await client.ping()}")
        
        print("\n=== 集群统计 ===")
        stats = ray.get(cluster.get_cluster_stats.remote())
        for stat in stats:
            print(f"Node {stat['node_id']}: {stat['keys_count']} keys")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # 清理资源
        client.close()
        ray.get(cluster.shutdown.remote())

if __name__ == "__main__":
    asyncio.run(main())
