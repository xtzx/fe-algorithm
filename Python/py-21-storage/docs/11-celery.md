# Celery 任务队列

> 分布式异步任务处理

## 什么是 Celery

Celery 是 Python 分布式任务队列，用于处理：
- 耗时任务（发邮件、生成报告）
- 定时任务（每日统计、清理缓存）
- 需要重试的任务（API 调用、支付）

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Producer │ ──▶ │  Broker  │ ──▶ │  Worker  │
│ (Web App)│     │ (Redis)  │     │ (Celery) │
└──────────┘     └──────────┘     └──────────┘
                       │
                       ▼
                ┌──────────┐
                │  Backend │
                │ (结果存储)│
                └──────────┘
```

---

## 安装配置

### 安装

```bash
pip install celery[redis]
# 或
pip install celery redis
```

### 基础配置

```python
# celery_app.py
from celery import Celery

app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1',
)

# 配置
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,

    # 任务默认配置
    task_acks_late=True,           # 任务完成后才确认
    task_reject_on_worker_lost=True,

    # Worker 配置
    worker_prefetch_multiplier=1,   # 一次取一个任务
    worker_concurrency=4,           # 并发数
)
```

### 项目结构

```
myproject/
├── celery_app.py      # Celery 实例
├── tasks/
│   ├── __init__.py
│   ├── email.py       # 邮件任务
│   └── report.py      # 报告任务
├── config.py          # 配置
└── main.py            # FastAPI
```

---

## 定义任务

### 基础任务

```python
# tasks/email.py
from celery_app import app

@app.task
def send_email(to: str, subject: str, body: str):
    """发送邮件"""
    # 模拟发送
    import time
    time.sleep(5)
    print(f"Email sent to {to}")
    return {"status": "sent", "to": to}
```

### 带参数的任务

```python
@app.task(
    bind=True,              # 第一个参数是 self
    max_retries=3,          # 最大重试次数
    default_retry_delay=60, # 默认重试延迟（秒）
)
def send_notification(self, user_id: int, message: str):
    try:
        # 发送通知
        notify_user(user_id, message)
    except ConnectionError as e:
        # 重试
        raise self.retry(exc=e)
```

### 任务选项

```python
@app.task(
    name='tasks.process_order',   # 自定义任务名
    queue='high_priority',        # 指定队列
    rate_limit='10/m',            # 速率限制
    time_limit=300,               # 硬超时（秒）
    soft_time_limit=240,          # 软超时
    ignore_result=False,          # 是否保存结果
    acks_late=True,               # 任务完成后确认
)
def process_order(order_id: int):
    pass
```

---

## 调用任务

### 异步调用

```python
from tasks.email import send_email

# 异步调用
result = send_email.delay("user@example.com", "Hello", "Body")

# 等同于
result = send_email.apply_async(
    args=["user@example.com", "Hello", "Body"]
)

# 获取任务 ID
print(result.id)

# 检查状态
print(result.status)  # PENDING, STARTED, SUCCESS, FAILURE

# 获取结果（阻塞）
print(result.get(timeout=10))

# 非阻塞检查
if result.ready():
    print(result.result)
```

### 延迟执行

```python
from datetime import datetime, timedelta

# 延迟 10 秒
send_email.apply_async(
    args=["user@example.com", "Hello", "Body"],
    countdown=10
)

# 指定时间
eta = datetime.now() + timedelta(hours=1)
send_email.apply_async(
    args=["user@example.com", "Hello", "Body"],
    eta=eta
)
```

### 任务链

```python
from celery import chain, group, chord

# 链：顺序执行
chain(
    task1.s(arg1),      # .s() 创建签名
    task2.s(),          # 上一个结果作为参数
    task3.s()
)()

# 组：并行执行
group(
    task1.s(1),
    task1.s(2),
    task1.s(3)
)()

# 和弦：并行 + 回调
chord(
    group(task1.s(1), task1.s(2), task1.s(3)),
    combine_results.s()  # 收集所有结果
)()
```

---

## 定时任务（Celery Beat）

### 配置定时任务

```python
# celery_app.py
from celery.schedules import crontab

app.conf.beat_schedule = {
    # 每分钟执行
    'check-every-minute': {
        'task': 'tasks.check_status',
        'schedule': 60.0,  # 秒
    },

    # 每天凌晨 2 点
    'daily-cleanup': {
        'task': 'tasks.cleanup',
        'schedule': crontab(hour=2, minute=0),
    },

    # 每周一 9 点
    'weekly-report': {
        'task': 'tasks.weekly_report',
        'schedule': crontab(hour=9, minute=0, day_of_week=1),
        'args': ('weekly',),
    },

    # 每月 1 号
    'monthly-billing': {
        'task': 'tasks.monthly_billing',
        'schedule': crontab(hour=0, minute=0, day_of_month=1),
    },
}
```

### 运行 Beat

```bash
# 启动 Beat（定时调度）
celery -A celery_app beat --loglevel=info

# 启动 Worker（执行任务）
celery -A celery_app worker --loglevel=info

# 或一起启动（开发环境）
celery -A celery_app worker --beat --loglevel=info
```

---

## 重试与错误处理

### 自动重试

```python
@app.task(
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_kwargs={'max_retries': 3, 'countdown': 5},
    retry_backoff=True,      # 指数退避
    retry_backoff_max=600,   # 最大延迟
    retry_jitter=True,       # 随机抖动
)
def unreliable_task(self, url: str):
    response = requests.get(url, timeout=10)
    return response.json()
```

### 手动重试

```python
@app.task(bind=True, max_retries=3)
def process_payment(self, order_id: int):
    try:
        result = payment_gateway.charge(order_id)
        return result
    except PaymentError as e:
        # 特定错误重试
        if e.is_retriable:
            raise self.retry(exc=e, countdown=60)
        else:
            # 不可重试的错误
            raise
```

### 错误回调

```python
from celery_app import app

@app.task
def on_task_failure(request, exc, traceback):
    """任务失败回调"""
    print(f"Task {request.id} failed: {exc}")

# 使用
send_email.apply_async(
    args=["user@example.com", "Hello", "Body"],
    link_error=on_task_failure.s()
)
```

---

## 与 FastAPI 集成

### 项目结构

```python
# main.py
from fastapi import FastAPI, BackgroundTasks
from tasks.email import send_email

app = FastAPI()

@app.post("/send-email")
async def send_email_endpoint(to: str, subject: str, body: str):
    # 提交到 Celery
    task = send_email.delay(to, subject, body)
    return {"task_id": task.id}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    from celery.result import AsyncResult
    result = AsyncResult(task_id)

    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }
```

### 取消任务

```python
from celery.result import AsyncResult

@app.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    result = AsyncResult(task_id)
    result.revoke(terminate=True)
    return {"status": "cancelled"}
```

---

## 监控（Flower）

### 安装

```bash
pip install flower
```

### 运行

```bash
celery -A celery_app flower --port=5555
```

### 访问

打开 `http://localhost:5555` 查看：
- 活跃 Worker
- 任务状态
- 队列长度
- 成功/失败率

---

## 生产环境配置

### 配置示例

```python
# config.py
import os

class Config:
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']

    # 任务路由
    CELERY_TASK_ROUTES = {
        'tasks.email.*': {'queue': 'email'},
        'tasks.report.*': {'queue': 'report'},
        'tasks.high_priority.*': {'queue': 'high'},
    }

    # 队列配置
    CELERY_TASK_QUEUES = {
        'default': {},
        'email': {},
        'report': {},
        'high': {},
    }
```

### 多队列 Worker

```bash
# 只处理 email 队列
celery -A celery_app worker -Q email --concurrency=2

# 处理多个队列
celery -A celery_app worker -Q default,email,high --concurrency=4

# 高优先级队列单独处理
celery -A celery_app worker -Q high --concurrency=8
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  celery-worker:
    build: .
    command: celery -A celery_app worker --loglevel=info
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    deploy:
      replicas: 2

  celery-beat:
    build: .
    command: celery -A celery_app beat --loglevel=info
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  flower:
    build: .
    command: celery -A celery_app flower --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 传递大对象 | 序列化慢/失败 | 只传 ID，任务内查询 |
| 任务内数据库连接 | 连接泄漏 | 使用连接池/每次新建 |
| 忘记启动 Worker | 任务不执行 | 确保 Worker 运行 |
| 结果过期 | 找不到结果 | 配置 result_expires |
| 任务幂等性 | 重试导致重复执行 | 设计幂等任务 |

---

## 小结

1. **定义任务**：`@app.task` 装饰器
2. **调用方式**：`delay()` / `apply_async()`
3. **定时任务**：`beat_schedule` + `celery beat`
4. **重试机制**：`autoretry_for` / `self.retry()`
5. **监控**：Flower Web UI
6. **生产部署**：多 Worker + 多队列

