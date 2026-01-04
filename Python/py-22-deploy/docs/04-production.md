# 生产实践

## 1. 优雅停机

### 1.1 什么是优雅停机

优雅停机确保：
- 当前请求处理完成
- 资源正确释放
- 数据不会丢失

### 1.2 FastAPI 实现

```python
import signal
import sys
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    print("Starting up...")
    await init_resources()
    
    yield
    
    # 关闭
    print("Shutting down...")
    await cleanup_resources()

app = FastAPI(lifespan=lifespan)

# 信号处理
def handle_shutdown(signum, frame):
    print(f"Received signal {signum}")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)
```

### 1.3 Gunicorn 优雅停机

```bash
# 优雅停机配置
gunicorn main:app \
    --timeout 120 \        # 请求超时
    --graceful-timeout 30  # 优雅停机等待时间

# 发送 SIGTERM 触发优雅停机
kill -SIGTERM <pid>

# 重新加载（零停机）
kill -SIGHUP <pid>
```

### 1.4 Docker 优雅停机

```dockerfile
# 正确传递信号
CMD ["gunicorn", "main:app", ...]  # exec 形式

# 设置停机超时
STOPSIGNAL SIGTERM
```

```bash
# Docker 停止命令
docker stop --time 30 container_name
```

## 2. 配置管理

### 2.1 分层配置

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 默认值
    app_env: str = "development"
    debug: bool = False
    
    # 数据库
    database_url: str = "sqlite:///./app.db"
    
    # 从 .env 文件加载
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 环境变量覆盖
# APP_ENV=production DATABASE_URL=postgresql://... python main.py
```

### 2.2 环境隔离

```
config/
├── base.py        # 公共配置
├── development.py # 开发配置
├── staging.py     # 测试配置
└── production.py  # 生产配置
```

```python
import os

env = os.getenv("APP_ENV", "development")
config = importlib.import_module(f"config.{env}")
```

### 2.3 12-Factor App 原则

1. 配置存储在环境变量中
2. 不同环境使用相同代码
3. 敏感信息不提交到代码库

## 3. 密钥管理

### 3.1 开发环境

```bash
# .env 文件（不提交到 Git）
SECRET_KEY=dev-secret-key-not-for-production
DATABASE_URL=postgresql://localhost/myapp
```

### 3.2 生产环境

```python
# 方式 1: 环境变量
SECRET_KEY = os.environ["SECRET_KEY"]

# 方式 2: 密钥管理服务
# AWS Secrets Manager / HashiCorp Vault / Azure Key Vault

# 方式 3: Docker Secrets
# docker secret create my_secret secret.txt
```

### 3.3 密钥轮换

```python
# 支持多个密钥进行轮换
SECRET_KEYS = os.environ["SECRET_KEYS"].split(",")
CURRENT_KEY = SECRET_KEYS[0]
OLD_KEYS = SECRET_KEYS[1:]

def verify_signature(data, signature):
    # 先用当前密钥验证
    if verify_with_key(data, signature, CURRENT_KEY):
        return True
    # 再用旧密钥验证
    for key in OLD_KEYS:
        if verify_with_key(data, signature, key):
            return True
    return False
```

## 4. CI/CD

### 4.1 GitHub Actions 示例

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: myregistry/myapp:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # kubectl set image deployment/myapp myapp=myregistry/myapp:${{ github.sha }}
          echo "Deploying..."
```

### 4.2 部署策略

| 策略 | 描述 | 风险 |
|------|------|------|
| 滚动更新 | 逐步替换实例 | 低 |
| 蓝绿部署 | 两套环境切换 | 低 |
| 金丝雀部署 | 小流量测试 | 最低 |
| 直接替换 | 停机更新 | 高 |

## 5. 故障处理

### 5.1 熔断器

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
async def call_external_service():
    async with httpx.AsyncClient() as client:
        return await client.get("https://api.example.com")
```

### 5.2 重试策略

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
)
async def unreliable_operation():
    ...
```

### 5.3 超时控制

```python
import asyncio

async def with_timeout(coro, timeout=10):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Operation timed out")
        raise
```

## 6. 性能优化

### 6.1 连接池

```python
# 数据库连接池
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# HTTP 客户端连接池
limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
client = httpx.AsyncClient(limits=limits)
```

### 6.2 缓存策略

```python
# 函数结果缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(arg):
    ...

# Redis 缓存
async def get_user(user_id: int):
    cached = await redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    user = await db.get_user(user_id)
    await redis.setex(f"user:{user_id}", 300, json.dumps(user))
    return user
```


