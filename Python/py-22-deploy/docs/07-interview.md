# 面试高频题

## 1. uvicorn 和 gunicorn 的区别？

### 答案

| 特性 | Uvicorn | Gunicorn |
|------|---------|----------|
| 协议 | ASGI | WSGI（通过 worker 支持 ASGI）|
| 异步 | 原生支持 | 需要 uvicorn worker |
| 多进程 | 有限支持 | 完善的进程管理 |
| 热重载 | 支持 | 需要 worker reload |
| 适用场景 | 开发、单进程 | 生产多进程 |

**生产推荐**：

```bash
gunicorn main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000
```

Gunicorn 负责进程管理，Uvicorn worker 处理异步请求。

---

## 2. 如何实现优雅停机？

### 答案

**优雅停机流程**：
1. 停止接收新请求
2. 等待当前请求完成
3. 释放资源（关闭连接）
4. 退出

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    await init_resources()
    yield
    # 关闭
    await cleanup_resources()

app = FastAPI(lifespan=lifespan)

# Gunicorn 配置
# --graceful-timeout 30  # 等待请求完成的时间
```

**关键配置**：
- Docker: `STOPSIGNAL SIGTERM`
- Kubernetes: `terminationGracePeriodSeconds`
- Gunicorn: `graceful_timeout`

---

## 3. 如何设计健康检查？

### 答案

**两种检查**：

1. **存活检查（Liveness）**
   - 目的：检查应用是否在运行
   - 失败后果：重启容器
   - 实现：简单返回 200

2. **就绪检查（Readiness）**
   - 目的：检查是否准备好接收流量
   - 失败后果：从负载均衡移除
   - 实现：检查所有依赖

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/health/ready")
async def ready():
    db_ok = await check_database()
    cache_ok = await check_cache()
    
    if not (db_ok and cache_ok):
        raise HTTPException(503, "Not ready")
    return {"status": "ready"}
```

**最佳实践**：
- 存活检查要简单快速
- 就绪检查要检查关键依赖
- 设置合理的超时和重试

---

## 4. Docker 多阶段构建的作用？

### 答案

**优势**：
1. **更小的镜像** - 不包含构建工具
2. **更安全** - 减少攻击面
3. **更快部署** - 传输和启动更快

```dockerfile
# 构建阶段（包含 gcc、make 等）
FROM python:3.11-slim as builder
RUN apt-get install -y build-essential
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 运行阶段（只有运行时依赖）
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . .
```

**效果**：镜像从 900MB 降到 150MB

---

## 5. 如何管理生产环境配置？

### 答案

**12-Factor App 原则**：配置存储在环境变量中

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**分层策略**：
1. 代码中定义默认值（非敏感）
2. `.env` 文件覆盖（开发环境）
3. 环境变量覆盖（生产环境）

**敏感信息**：
- 使用密钥管理服务（Vault、AWS Secrets Manager）
- 不要提交到代码库
- 不要有默认值

---

## 6. 如何收集和分析日志？

### 答案

**结构化日志**：

```python
import structlog

logger = structlog.get_logger()
logger.info("request_handled", 
    method="GET", 
    path="/api/users",
    duration_ms=42)

# 输出 JSON
# {"event": "request_handled", "method": "GET", "path": "/api/users", "duration_ms": 42}
```

**日志聚合方案**：
1. **ELK Stack** - Elasticsearch + Logstash + Kibana
2. **Loki + Grafana** - 轻量级方案
3. **云服务** - CloudWatch、Stackdriver

**最佳实践**：
- 使用结构化日志（JSON）
- 包含 request_id 用于追踪
- 设置合理的日志级别
- 定期清理历史日志

---

## 7. 什么是分布式追踪？

### 答案

**概念**：追踪一个请求在分布式系统中的完整路径。

**组成**：
- **Trace**：一次完整请求
- **Span**：单个操作
- **Context**：跨服务传播的信息

```
Trace: 用户请求
├── Span: API Gateway (50ms)
├── Span: User Service (30ms)
│   └── Span: Database Query (20ms)
└── Span: Order Service (40ms)
    ├── Span: Cache Lookup (5ms)
    └── Span: Message Queue (10ms)
```

**工具**：
- OpenTelemetry（标准）
- Jaeger、Zipkin（收集和展示）

**用途**：
- 发现性能瓶颈
- 排查分布式问题
- 理解服务依赖

---

## 8. 如何监控 Python 服务的性能？

### 答案

**指标类型**：
1. **RED 方法**（面向服务）
   - Rate（请求率）
   - Errors（错误率）
   - Duration（延迟）

2. **USE 方法**（面向资源）
   - Utilization（利用率）
   - Saturation（饱和度）
   - Errors（错误数）

**Prometheus 指标**：

```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("http_requests_total", "Total requests", 
    ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Latency",
    ["method", "endpoint"])
```

**监控工具栈**：
- 指标：Prometheus + Grafana
- 日志：Loki / ELK
- 追踪：Jaeger / Zipkin
- APM：Datadog / New Relic

**关键指标**：
- P50/P95/P99 延迟
- 错误率
- QPS
- 资源使用（CPU、内存）

---

## 附加题

### 9. 容器化部署 vs 传统部署的优缺点？

**容器化优势**：
- 环境一致性
- 快速部署
- 资源隔离
- 易于扩展

**容器化挑战**：
- 学习成本
- 调试复杂
- 网络配置
- 持久化存储

### 10. 如何实现零停机部署？

**策略**：

1. **滚动更新**
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1
    maxSurge: 1
```

2. **蓝绿部署**
- 两套环境同时运行
- 切换流量

3. **金丝雀部署**
- 小流量测试新版本
- 逐步增加流量


