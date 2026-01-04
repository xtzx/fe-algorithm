# Docker 部署

## 1. Dockerfile 基础

### 1.1 简单 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 1.2 多阶段构建

```dockerfile
# 构建阶段
FROM python:3.11-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 运行阶段
FROM python:3.11-slim

WORKDIR /app

# 复制依赖
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**优势**：
- 更小的镜像体积
- 不包含构建工具
- 更安全

## 2. Dockerfile 最佳实践

### 2.1 层缓存优化

```dockerfile
# ✅ 好：先复制依赖文件
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# ❌ 坏：每次代码变更都重新安装依赖
COPY . .
RUN pip install -r requirements.txt
```

### 2.2 使用 .dockerignore

`.dockerignore`:

```
.git
.gitignore
__pycache__
*.pyc
*.pyo
.env
.venv
venv
*.egg-info
.pytest_cache
.coverage
htmlcov
.mypy_cache
```

### 2.3 非 root 用户

```dockerfile
# 创建用户
RUN useradd --create-home --shell /bin/bash appuser

# 切换用户
USER appuser

# 或使用 nobody
USER nobody
```

### 2.4 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

## 3. Docker Compose

### 3.1 基础配置

```yaml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
    env_file:
      - .env
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

### 3.2 健康检查

```yaml
services:
  app:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    depends_on:
      db:
        condition: service_healthy

  db:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### 3.3 资源限制

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
        reservations:
          cpus: "0.25"
          memory: 256M
```

## 4. 环境变量管理

### 4.1 方式对比

```yaml
services:
  app:
    # 方式 1: 直接定义
    environment:
      - APP_ENV=production
      - DEBUG=false
    
    # 方式 2: 使用文件
    env_file:
      - .env
      - .env.production
    
    # 方式 3: 从主机环境
    environment:
      - SECRET_KEY  # 继承主机的 SECRET_KEY
```

### 4.2 敏感信息处理

```bash
# 不要将敏感信息写入 Dockerfile 或提交到 Git
# 使用运行时注入

# 方式 1: 运行时传入
docker run -e SECRET_KEY=xxx myapp

# 方式 2: 使用 Docker secrets（Swarm）
# 方式 3: 使用外部密钥管理（Vault、AWS Secrets Manager）
```

## 5. 常用命令

```bash
# 构建镜像
docker build -t myapp:latest .

# 运行容器
docker run -d -p 8000:8000 --name myapp myapp:latest

# 查看日志
docker logs -f myapp

# 进入容器
docker exec -it myapp /bin/bash

# 停止并删除
docker stop myapp && docker rm myapp

# Docker Compose
docker-compose up -d      # 启动
docker-compose down       # 停止
docker-compose logs -f    # 日志
docker-compose ps         # 状态
docker-compose restart    # 重启
```

## 6. 镜像优化

### 6.1 选择合适的基础镜像

| 镜像 | 大小 | 适用场景 |
|------|------|----------|
| python:3.11 | ~900MB | 开发、调试 |
| python:3.11-slim | ~150MB | 生产环境 |
| python:3.11-alpine | ~50MB | 极致精简 |

### 6.2 Alpine 注意事项

```dockerfile
# Alpine 使用 musl libc，可能有兼容性问题
FROM python:3.11-alpine

# 需要额外安装编译依赖
RUN apk add --no-cache gcc musl-dev
```

## 7. CI/CD 集成

```yaml
# .github/workflows/docker.yml
name: Docker Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: myregistry/myapp:latest
```


