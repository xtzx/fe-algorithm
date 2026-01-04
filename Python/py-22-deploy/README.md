# P22: 部署与可观测性

> 把服务部署到生产环境

## 🎯 学完后能做

- 使用 Docker 部署
- 配置生产级日志
- 实现健康检查

## 📁 目录结构

```
py-22-deploy/
├── README.md
├── docs/
│   ├── 01-asgi-server.md        # ASGI 服务器
│   ├── 02-docker.md             # Docker 部署
│   ├── 03-observability.md      # 可观测性
│   ├── 04-production.md         # 生产实践
│   ├── 05-distribution.md       # 脚本分发
│   ├── 06-exercises.md          # 练习题
│   └── 07-interview.md          # 面试题
├── examples/
│   ├── docker/
│   │   ├── Dockerfile           # 多阶段构建示例
│   │   ├── Dockerfile.simple    # 简单构建示例
│   │   ├── docker-compose.yml   # 完整服务编排
│   │   └── .env.example         # 环境变量示例
│   ├── zipapp_demo/
│   │   ├── __main__.py          # 入口文件
│   │   ├── app.py               # 应用代码
│   │   └── build.sh             # 构建脚本
│   └── observability/
│       ├── logging_config.py    # 日志配置
│       ├── metrics.py           # Prometheus 指标
│       ├── tracing.py           # 分布式追踪
│       └── health.py            # 健康检查
└── scripts/
    ├── build_docker.sh          # Docker 构建
    ├── build_zipapp.sh          # ZipApp 构建
    └── run_prod.sh              # 生产运行
```

## 🚀 快速开始

### Docker 部署

```bash
cd examples/docker

# 构建镜像
docker build -t myapp:latest .

# 运行容器
docker run -d -p 8000:8000 --env-file .env myapp:latest

# 或使用 Docker Compose
docker-compose up -d
```

### 生产运行（无 Docker）

```bash
# 使用 gunicorn + uvicorn workers
gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30
```

### ZipApp 分发

```bash
cd examples/zipapp_demo
./build.sh

# 运行
python myapp.pyz
```

## 🔧 核心概念

### 1. ASGI 服务器

```bash
# 开发模式
uvicorn main:app --reload

# 生产模式（单进程）
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

# 生产模式（多进程）
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 2. Docker 多阶段构建

```dockerfile
# 构建阶段
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 运行阶段
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. 结构化日志

```python
import structlog

logger = structlog.get_logger()
logger.info("request_handled", method="GET", path="/api/users", duration_ms=42)
```

### 4. 健康检查

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/health/ready")
async def readiness():
    # 检查数据库、Redis 等依赖
    db_ok = await check_database()
    cache_ok = await check_cache()
    
    if db_ok and cache_ok:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Not ready")
```

## 📚 学习路径

1. **ASGI 服务器** - uvicorn、gunicorn
2. **Docker** - Dockerfile、Compose
3. **可观测性** - 日志、指标、追踪
4. **生产实践** - 优雅停机、配置管理
5. **脚本分发** - zipapp、pex

## ✅ 功能清单

- [x] uvicorn 配置
- [x] gunicorn + uvicorn workers
- [x] 进程管理
- [x] Dockerfile 编写
- [x] 多阶段构建
- [x] Docker Compose
- [x] 环境变量管理
- [x] 结构化日志
- [x] Prometheus metrics（概念）
- [x] OpenTelemetry tracing（概念）
- [x] 健康检查端点
- [x] 优雅停机
- [x] 配置管理
- [x] 密钥管理
- [x] CI/CD 概念
- [x] zipapp


