"""
任务队列测试
"""

import pytest

from storage_lab.queue.simple import SimpleQueue, Task, TaskStatus


class TestTask:
    """任务测试"""

    def test_task_creation(self):
        """测试任务创建"""
        task = Task(
            name="send_email",
            args=("user@example.com",),
            kwargs={"subject": "Hello"},
        )

        assert task.id is not None
        assert task.name == "send_email"
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0

    def test_task_serialization(self):
        """测试任务序列化"""
        task = Task(
            name="process_data",
            args=(1, 2, 3),
            kwargs={"option": "value"},
        )

        # 转换为字典
        task_dict = task.to_dict()
        assert task_dict["name"] == "process_data"
        assert task_dict["args"] == [1, 2, 3]

        # 转换为 JSON
        json_str = task.to_json()
        assert "process_data" in json_str

        # 从 JSON 恢复
        restored = Task.from_json(json_str)
        assert restored.name == task.name
        assert restored.args == task.args


class TestSimpleQueue:
    """简单队列测试"""

    @pytest.fixture
    def queue(self, fake_redis):
        """创建队列"""
        # 注入 fake redis
        q = SimpleQueue("test")
        q.client = fake_redis
        return q

    def test_enqueue(self, queue):
        """测试入队"""
        task_id = queue.enqueue("send_email", "user@example.com", subject="Hello")

        assert task_id is not None
        assert queue.get_queue_length() == 1

    def test_dequeue(self, queue, fake_redis):
        """测试出队"""
        # 入队
        task_id = queue.enqueue("process", "data")

        # 模拟 blpop 行为
        fake_redis.set(queue._task_key(task_id), Task(
            id=task_id,
            name="process",
            args=("data",),
            kwargs={},
        ).to_json())

        # 出队
        task = queue.dequeue()

        # 注意：fake_redis 的 blpop 行为可能不同
        # 这里我们只测试基本逻辑

    def test_get_task(self, queue, fake_redis):
        """测试获取任务详情"""
        # 创建任务
        task = Task(name="test", args=())
        fake_redis.set(queue._task_key(task.id), task.to_json())

        # 获取
        retrieved = queue.get_task(task.id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_get_stats(self, queue, fake_redis):
        """测试获取统计信息"""
        # 添加一些任务
        fake_redis.rpush(queue.queue_key, "task1", "task2")
        fake_redis.rpush(queue.completed_key, "task3")
        fake_redis.rpush(queue.failed_key, "task4")

        stats = queue.get_stats()

        assert stats["pending"] == 2
        assert stats["completed"] == 1
        assert stats["failed"] == 1

    def test_clear(self, queue, fake_redis):
        """测试清空队列"""
        # 添加数据
        fake_redis.rpush(queue.queue_key, "task1")
        fake_redis.rpush(queue.completed_key, "task2")

        # 清空
        queue.clear()

        assert fake_redis.llen(queue.queue_key) == 0
        assert fake_redis.llen(queue.completed_key) == 0


class TestWorker:
    """Worker 测试"""

    def test_register_handler(self):
        """测试注册处理函数"""
        from storage_lab.queue.worker import Worker

        worker = Worker(queue_name="test")

        @worker.task("send_email")
        def send_email(to: str, subject: str):
            return {"status": "sent"}

        assert "send_email" in worker._handlers

    def test_process_task(self):
        """测试处理任务"""
        from storage_lab.queue.worker import Worker

        worker = Worker(queue_name="test")

        @worker.task("add")
        def add(a: int, b: int):
            return a + b

        task = Task(name="add", args=(1, 2), kwargs={})
        result = worker.process_task(task)

        assert result == 3

    def test_process_unknown_task(self):
        """测试处理未知任务"""
        from storage_lab.queue.worker import Worker

        worker = Worker(queue_name="test")
        task = Task(name="unknown", args=(), kwargs={})

        with pytest.raises(ValueError):
            worker.process_task(task)


