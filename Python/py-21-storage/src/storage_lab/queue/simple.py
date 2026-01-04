"""
简单任务队列

使用 Redis List 实现基本的任务队列

功能:
- 任务入队
- 任务出队
- 任务状态追踪
- 失败重试
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from storage_lab.cache.client import CacheClient, get_cache_client


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"      # 等待执行
    PROCESSING = "processing"  # 正在执行
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"        # 执行失败
    RETRY = "retry"          # 等待重试


@dataclass
class Task:
    """
    任务数据类
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "args": list(self.args),
            "kwargs": self.kwargs,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            args=tuple(data.get("args", [])),
            kwargs=data.get("kwargs", {}),
            status=TaskStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, data: str) -> "Task":
        """从 JSON 反序列化"""
        return cls.from_dict(json.loads(data))


class SimpleQueue:
    """
    简单任务队列
    
    使用 Redis List 实现 FIFO 队列
    
    Usage:
        queue = SimpleQueue("tasks")
        
        # 添加任务
        task_id = queue.enqueue("send_email", "user@example.com", subject="Hello")
        
        # 获取任务
        task = queue.dequeue()
        
        # 处理任务
        if task:
            result = process_task(task)
            queue.complete(task, result)
    """
    
    def __init__(
        self,
        name: str = "default",
        client: Optional[CacheClient] = None,
    ):
        self.name = name
        self.client = client or get_cache_client()
        
        # Redis 键
        self.queue_key = f"queue:{name}:pending"
        self.processing_key = f"queue:{name}:processing"
        self.completed_key = f"queue:{name}:completed"
        self.failed_key = f"queue:{name}:failed"
        self.task_prefix = f"queue:{name}:task"
    
    def _task_key(self, task_id: str) -> str:
        """获取任务详情的键"""
        return f"{self.task_prefix}:{task_id}"
    
    def enqueue(
        self,
        task_name: str,
        *args,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """
        将任务加入队列
        
        Args:
            task_name: 任务名称
            *args: 任务参数
            max_retries: 最大重试次数
            **kwargs: 任务关键字参数
        
        Returns:
            任务 ID
        """
        task = Task(
            name=task_name,
            args=args,
            kwargs=kwargs,
            max_retries=max_retries,
        )
        
        # 保存任务详情
        self.client.set(self._task_key(task.id), task.to_json(), ttl=86400)
        
        # 加入队列
        self.client.rpush(self.queue_key, task.id)
        
        return task.id
    
    def dequeue(self, timeout: int = 0) -> Optional[Task]:
        """
        从队列取出任务
        
        Args:
            timeout: 阻塞超时时间（秒），0 表示不阻塞
        
        Returns:
            任务对象，如果队列为空返回 None
        """
        if timeout > 0:
            # 阻塞式获取
            result = self.client.client.blpop(self.queue_key, timeout=timeout)
            if result is None:
                return None
            task_id = result[1]
        else:
            # 非阻塞获取
            task_id = self.client.lpop(self.queue_key)
            if task_id is None:
                return None
        
        # 获取任务详情
        task_data = self.client.get(self._task_key(task_id))
        if task_data is None:
            return None
        
        task = Task.from_json(task_data)
        
        # 更新状态
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()
        self.client.set(self._task_key(task.id), task.to_json(), ttl=86400)
        
        # 加入处理中队列
        self.client.rpush(self.processing_key, task.id)
        
        return task
    
    def complete(self, task: Task, result: Any = None):
        """
        标记任务完成
        
        Args:
            task: 任务对象
            result: 任务结果
        """
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = time.time()
        
        # 更新任务详情
        self.client.set(self._task_key(task.id), task.to_json(), ttl=86400)
        
        # 从处理中队列移除
        self.client.client.lrem(self.processing_key, 1, task.id)
        
        # 加入完成队列
        self.client.rpush(self.completed_key, task.id)
    
    def fail(self, task: Task, error: str):
        """
        标记任务失败
        
        如果未达到最大重试次数，重新入队
        
        Args:
            task: 任务对象
            error: 错误信息
        """
        task.error = error
        task.retry_count += 1
        
        # 从处理中队列移除
        self.client.client.lrem(self.processing_key, 1, task.id)
        
        if task.retry_count < task.max_retries:
            # 重试
            task.status = TaskStatus.RETRY
            self.client.set(self._task_key(task.id), task.to_json(), ttl=86400)
            self.client.rpush(self.queue_key, task.id)
        else:
            # 最终失败
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            self.client.set(self._task_key(task.id), task.to_json(), ttl=86400)
            self.client.rpush(self.failed_key, task.id)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务详情"""
        task_data = self.client.get(self._task_key(task_id))
        if task_data:
            return Task.from_json(task_data)
        return None
    
    def get_queue_length(self) -> int:
        """获取待处理队列长度"""
        return self.client.llen(self.queue_key)
    
    def get_processing_count(self) -> int:
        """获取正在处理的任务数"""
        return self.client.llen(self.processing_key)
    
    def get_stats(self) -> dict:
        """获取队列统计信息"""
        return {
            "pending": self.client.llen(self.queue_key),
            "processing": self.client.llen(self.processing_key),
            "completed": self.client.llen(self.completed_key),
            "failed": self.client.llen(self.failed_key),
        }
    
    def clear(self):
        """清空队列"""
        self.client.delete(self.queue_key)
        self.client.delete(self.processing_key)
        self.client.delete(self.completed_key)
        self.client.delete(self.failed_key)


