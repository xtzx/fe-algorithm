# 🐳 Docker 部署

> 容器化 LLM 服务

---

## Docker 基础

### Dockerfile 编写

```dockerfile
# Dockerfile - FastAPI 服务
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 优化的 Dockerfile（多阶段构建）

```dockerfile
# 构建阶段
FROM python:3.11-slim as builder

WORKDIR /app

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 运行阶段
FROM python:3.11-slim

WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制代码
COPY . .

# 非 root 用户
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### .dockerignore

```
# .dockerignore
__pycache__
*.pyc
*.pyo
.git
.gitignore
.env
.venv
venv
*.md
tests/
.pytest_cache
.coverage
htmlcov
.mypy_cache
*.egg-info
dist
build
```

---

## GPU 容器支持

### NVIDIA Container Toolkit 安装

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### GPU Dockerfile

```dockerfile
# GPU 版本 - vLLM 服务
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# 安装 Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 安装 vLLM
RUN pip3 install vllm

# 复制代码
COPY . .

EXPOSE 8000

# 启动 vLLM 服务
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen2.5-7B-Instruct", \
     "--host", "0.0.0.0", "--port", "8000"]
```

### 运行 GPU 容器

```bash
# 使用所有 GPU
docker run --gpus all -p 8000:8000 my-vllm-service

# 指定 GPU
docker run --gpus '"device=0,1"' -p 8000:8000 my-vllm-service

# 设置显存限制（通过环境变量）
docker run --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -p 8000:8000 my-vllm-service
```

---

## Docker Compose

### 基础配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=ollama:11434
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

### 完整生产配置

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - api
    restart: unless-stopped

  # API 服务
  api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - LLM_BACKEND=http://vllm:8000/v1
      - LOG_LEVEL=INFO
    depends_on:
      - vllm
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # vLLM 推理服务
  vllm:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --host 0.0.0.0
      --port 8000
      --gpu-memory-utilization 0.9
    volumes:
      - model_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # 向量数据库
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    restart: unless-stopped

  # 监控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  model_cache:
  chroma_data:
  grafana_data:
```

### Nginx 配置

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }

    # 限流配置
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    server {
        listen 80;
        server_name api.example.com;

        # 限流
        limit_req zone=api_limit burst=20 nodelay;

        location / {
            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            # SSE 支持
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 300s;
        }

        location /health {
            proxy_pass http://api_backend/health;
        }
    }
}
```

---

## 常用命令

```bash
# 构建
docker build -t my-llm-api .

# 运行
docker run -d -p 8000:8000 --name llm-api my-llm-api

# 查看日志
docker logs -f llm-api

# 进入容器
docker exec -it llm-api /bin/bash

# Docker Compose
docker compose up -d
docker compose logs -f
docker compose down

# 重新构建
docker compose up -d --build

# 查看资源使用
docker stats
```

---

## CI/CD 示例

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            myuser/llm-api:latest
            myuser/llm-api:${{ github.sha }}

      - name: Deploy to server
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            cd /app
            docker compose pull
            docker compose up -d
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - build
  - deploy

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy:
  stage: deploy
  script:
    - ssh user@server "cd /app && docker compose pull && docker compose up -d"
  only:
    - main
```

---

## 最佳实践

```
1. 镜像优化
   - 使用多阶段构建
   - 选择合适的基础镜像
   - 清理不必要的文件

2. 安全
   - 不使用 root 用户
   - 不在镜像中存储敏感信息
   - 使用 .dockerignore

3. 资源管理
   - 设置资源限制
   - 配置健康检查
   - 使用数据卷持久化

4. 日志
   - 输出到 stdout/stderr
   - 使用日志驱动收集
```

---

## ➡️ 下一步

继续 [09-监控与日志.md](./09-监控与日志.md)

