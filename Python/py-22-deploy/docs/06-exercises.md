# 练习题

## 练习 1: Dockerfile 编写

为以下 FastAPI 应用编写多阶段 Dockerfile：

```python
# main.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello"}
```

**要求**：
1. 使用多阶段构建
2. 使用非 root 用户
3. 添加健康检查
4. 镜像尽可能小

---

## 练习 2: Docker Compose

编写 docker-compose.yml 部署以下服务：
- FastAPI 应用（2 个副本）
- PostgreSQL 数据库
- Redis 缓存
- Nginx 反向代理

**要求**：
1. 使用健康检查确保启动顺序
2. 配置资源限制
3. 使用数据卷持久化

---

## 练习 3: 结构化日志

实现 FastAPI 请求日志中间件：

```python
# 要求输出：
# {"event": "request", "method": "GET", "path": "/api/users", 
#  "status": 200, "duration_ms": 45, "request_id": "xxx"}
```

**要求**：
1. 使用 structlog
2. 每个请求生成唯一 ID
3. 记录请求方法、路径、状态码、耗时
4. 支持 JSON 和控制台两种格式

---

## 练习 4: Prometheus 指标

为 API 添加以下指标：

1. `http_requests_total` - 请求总数（按方法、路径、状态码）
2. `http_request_duration_seconds` - 请求延迟分布
3. `http_requests_in_progress` - 当前进行中的请求数

**要求**：
1. 创建 `/metrics` 端点
2. 实现中间件自动记录

---

## 练习 5: 健康检查

实现完整的健康检查系统：

```python
# GET /health -> 存活检查
# GET /health/ready -> 就绪检查（检查 DB、Redis）
# GET /health/detail -> 详细健康报告
```

**要求**：
1. 存活检查只检查应用是否运行
2. 就绪检查检查所有依赖
3. 详细报告包含每个检查的耗时

---

## 练习 6: 优雅停机

实现支持优雅停机的应用：

**要求**：
1. 收到 SIGTERM 时停止接收新请求
2. 等待当前请求完成
3. 关闭数据库连接
4. 最多等待 30 秒

---

## 练习 7: 配置管理

实现分层配置系统：

```python
# 配置优先级：环境变量 > .env 文件 > 默认值
# 支持：development, staging, production 环境
```

**要求**：
1. 使用 pydantic-settings
2. 敏感信息不能有默认值
3. 支持类型验证

---

## 练习 8: ZipApp 打包

将以下 CLI 工具打包为 zipapp：

```python
# cli.py
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    print(f"Hello, {args.name}!")

if __name__ == "__main__":
    main()
```

**要求**：
1. 可以直接运行：`python myapp.pyz World`
2. 在 Unix 上可以 `./myapp.pyz World`

---

## 练习 9: Gunicorn 配置

编写 `gunicorn.conf.py` 配置文件：

**要求**：
1. 根据 CPU 核心数设置 workers
2. 配置优雅停机
3. 配置 worker 自动重启
4. 配置日志

---

## 练习 10: CI/CD 流水线

编写 GitHub Actions 工作流：

**要求**：
1. 运行测试
2. 构建 Docker 镜像
3. 推送到镜像仓库
4. 部署到服务器

---

## 参考答案

### 练习 1 答案

```dockerfile
# 构建阶段
FROM python:3.11-slim as builder
WORKDIR /app
RUN pip install --user fastapi uvicorn
COPY . .

# 运行阶段
FROM python:3.11-slim
WORKDIR /app
RUN useradd -m appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /app .
ENV PATH=/home/appuser/.local/bin:$PATH
USER appuser
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/ || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 练习 6 答案

```python
import asyncio
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI

shutdown_event = asyncio.Event()
active_requests = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    
    # 等待请求完成
    timeout = 30
    while active_requests > 0 and timeout > 0:
        await asyncio.sleep(1)
        timeout -= 1
    
    # 清理资源
    await close_db()
    await close_cache()

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def track_requests(request, call_next):
    global active_requests
    active_requests += 1
    try:
        return await call_next(request)
    finally:
        active_requests -= 1

def handle_shutdown(signum, frame):
    shutdown_event.set()

signal.signal(signal.SIGTERM, handle_shutdown)
```


