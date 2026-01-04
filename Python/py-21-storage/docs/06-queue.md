# 任务队列

## 概述

任务队列用于异步执行耗时操作：

1. **解耦** - 请求和处理分离
2. **削峰** - 平滑处理高峰流量
3. **重试** - 失败任务自动重试
4. **可靠** - 任务不丢失

## 1. 为什么需要任务队列

### 1.1 同步处理的问题

```python
@app.post("/orders")
def create_order(order: Order):
    # 1. 创建订单
    db.create_order(order)
    
    # 2. 发送邮件（耗时）
    send_email(order.user_email, "订单确认")
    
    # 3. 推送通知（耗时）
    push_notification(order.user_id, "订单已创建")
    
    # 4. 更新库存（可能失败）
    update_inventory(order.items)
    
    return {"status": "success"}
```

**问题**：
- 用户等待时间长
- 任何一步失败整个请求失败
- 无法重试失败的操作

### 1.2 使用任务队列

```python
@app.post("/orders")
def create_order(order: Order):
    # 1. 创建订单
    db.create_order(order)
    
    # 2. 异步执行其他操作
    queue.enqueue("send_email", order.user_email, "订单确认")
    queue.enqueue("push_notification", order.user_id, "订单已创建")
    queue.enqueue("update_inventory", order.items)
    
    return {"status": "success"}  # 立即返回
```

## 2. 简单实现（Redis）

### 2.1 任务队列

```python
import json
import uuid
from dataclasses import dataclass
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    name: str
    args: tuple
    kwargs: dict
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3

class SimpleQueue:
    def __init__(self, client, name="default"):
        self.client = client
        self.name = name
        self.queue_key = f"queue:{name}:pending"
    
    def enqueue(self, task_name: str, *args, **kwargs) -> str:
        """添加任务"""
        task = Task(
            id=str(uuid.uuid4()),
            name=task_name,
            args=args,
            kwargs=kwargs,
        )
        
        # 保存任务详情
        task_key = f"task:{task.id}"
        self.client.set(task_key, json.dumps(task.__dict__))
        
        # 加入队列
        self.client.rpush(self.queue_key, task.id)
        
        return task.id
    
    def dequeue(self, timeout: int = 0) -> Task | None:
        """获取任务"""
        result = self.client.blpop(self.queue_key, timeout=timeout)
        if result is None:
            return None
        
        task_id = result[1].decode()
        task_data = self.client.get(f"task:{task_id}")
        if task_data:
            return Task(**json.loads(task_data))
        return None
```

### 2.2 Worker

```python
class Worker:
    def __init__(self, queue: SimpleQueue):
        self.queue = queue
        self.handlers = {}
    
    def task(self, name: str):
        """任务装饰器"""
        def decorator(func):
            self.handlers[name] = func
            return func
        return decorator
    
    def process(self, task: Task):
        """处理任务"""
        handler = self.handlers.get(task.name)
        if handler is None:
            raise ValueError(f"Unknown task: {task.name}")
        return handler(*task.args, **task.kwargs)
    
    def run(self):
        """运行 Worker"""
        print(f"Worker started, listening on queue: {self.queue.name}")
        
        while True:
            task = self.queue.dequeue(timeout=5)
            if task is None:
                continue
            
            try:
                result = self.process(task)
                print(f"Task {task.id} completed: {result}")
            except Exception as e:
                print(f"Task {task.id} failed: {e}")
                # 重试逻辑...

# 使用
worker = Worker(SimpleQueue(redis_client))

@worker.task("send_email")
def send_email(to: str, subject: str):
    print(f"Sending email to {to}: {subject}")
    return {"status": "sent"}

worker.run()
```

## 3. RQ（Redis Queue）

```bash
pip install rq
```

### 3.1 定义任务

```python
# tasks.py
import time

def send_email(to: str, subject: str, body: str):
    """发送邮件任务"""
    time.sleep(2)  # 模拟耗时
    return {"to": to, "status": "sent"}

def generate_report(report_type: str, params: dict):
    """生成报告任务"""
    time.sleep(5)
    return {"type": report_type, "status": "generated"}
```

### 3.2 入队

```python
from redis import Redis
from rq import Queue

redis_conn = Redis()
q = Queue(connection=redis_conn)

# 入队
job = q.enqueue(send_email, "user@example.com", "Hello", "Body")

# 获取结果
job.result  # None（未完成）
job.is_finished  # False

# 等待完成
job.get_result(timeout=10)
```

### 3.3 运行 Worker

```bash
rq worker
```

## 4. Celery

```bash
pip install celery[redis]
```

### 4.1 配置

```python
# celery_app.py
from celery import Celery

app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1',
)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 分钟超时
)
```

### 4.2 定义任务

```python
# tasks.py
from celery_app import app

@app.task(bind=True, max_retries=3)
def send_email(self, to: str, subject: str, body: str):
    try:
        # 发送邮件逻辑
        return {"to": to, "status": "sent"}
    except Exception as exc:
        # 重试
        self.retry(exc=exc, countdown=60)

@app.task
def generate_report(report_type: str, params: dict):
    return {"type": report_type, "status": "generated"}
```

### 4.3 调用任务

```python
from tasks import send_email, generate_report

# 异步调用
result = send_email.delay("user@example.com", "Hello", "Body")

# 获取结果
result.ready()  # False
result.get(timeout=10)  # 阻塞等待

# 链式调用
from celery import chain
chain(task1.s(arg1), task2.s(), task3.s()).delay()

# 组调用
from celery import group
group(task.s(i) for i in range(10)).delay()
```

### 4.4 运行 Worker

```bash
celery -A celery_app worker --loglevel=info
```

## 5. 最佳实践

### 5.1 任务设计

```python
# ✅ 好：幂等任务
@app.task
def process_order(order_id: int):
    order = db.get_order(order_id)
    if order.status == "processed":
        return  # 已处理，跳过
    
    # 处理逻辑...
    order.status = "processed"
    db.save(order)

# ❌ 坏：非幂等任务
@app.task
def increment_counter(amount: int):
    counter = db.get_counter()
    counter.value += amount  # 重试会导致重复增加
    db.save(counter)
```

### 5.2 错误处理

```python
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def risky_task(self, data):
    try:
        return process(data)
    except TransientError as exc:
        # 可重试的错误
        raise self.retry(exc=exc)
    except PermanentError as exc:
        # 不可重试的错误
        logger.error(f"Task failed permanently: {exc}")
        raise
```

### 5.3 监控

```python
# 使用 Flower 监控 Celery
# pip install flower
# celery -A celery_app flower
```

## 对比

| 特性 | RQ | Celery |
|------|-----|--------|
| 复杂度 | 简单 | 复杂 |
| 功能 | 基础 | 丰富 |
| Broker | 仅 Redis | Redis/RabbitMQ |
| 定时任务 | 需要额外库 | 内置 |
| 监控 | rq-dashboard | Flower |
| 适用场景 | 小项目 | 大项目 |


