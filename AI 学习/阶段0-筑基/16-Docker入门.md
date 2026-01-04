# 🐳 16 - Docker 入门

> AI 开发必备：解决「在我电脑上能跑」的问题

---

## 目录

1. [为什么需要 Docker](#1-为什么需要-docker)
2. [核心概念](#2-核心概念)
3. [安装与配置](#3-安装与配置)
4. [基础命令](#4-基础命令)
5. [Dockerfile 编写](#5-dockerfile-编写)
6. [Docker Compose](#6-docker-compose)
7. [GPU 支持](#7-gpu-支持)
8. [实战：容器化 Python 项目](#8-实战容器化-python-项目)
9. [常见问题排查](#9-常见问题排查)

---

## 1. 为什么需要 Docker

### 1.1 环境一致性问题

```
开发场景常见的痛点：

👨‍💻 开发者 A: "代码在我电脑上能跑！"
👩‍💻 开发者 B: "在我这跑不起来，缺少 xxx 库"
🖥️ 服务器:    "版本不对，需要 Python 3.8 不是 3.11"
☁️ 云端:       "CUDA 版本不匹配..."

原因：
- Python 版本不同（3.8 vs 3.10 vs 3.11）
- 系统库版本不同（glibc, OpenSSL）
- 依赖包版本冲突（torch 1.x vs 2.x）
- 环境变量不同
- 操作系统差异（Ubuntu vs CentOS vs macOS）
```

### 1.2 Docker 如何解决

```
Docker 的解决方案：把整个运行环境打包！

┌─────────────────────────────────────┐
│           Docker 容器               │
│  ┌─────────────────────────────┐   │
│  │     你的应用代码             │   │
│  ├─────────────────────────────┤   │
│  │  Python 3.10 + 所有依赖包    │   │
│  ├─────────────────────────────┤   │
│  │  系统库（精确版本）          │   │
│  ├─────────────────────────────┤   │
│  │  Ubuntu 22.04 基础系统       │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘

在任何机器上运行这个容器，结果都一样！
```

### 1.3 Docker vs 虚拟机

```
┌─────────────────────────────────────────────────────────┐
│                    虚拟机 (VM)                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   App    │  │   App    │  │   App    │              │
│  ├──────────┤  ├──────────┤  ├──────────┤              │
│  │  Guest   │  │  Guest   │  │  Guest   │              │
│  │    OS    │  │    OS    │  │    OS    │  ← 每个都要完整 OS │
│  └──────────┘  └──────────┘  └──────────┘              │
│  ┌───────────────────────────────────────┐              │
│  │            Hypervisor                  │              │
│  └───────────────────────────────────────┘              │
│  ┌───────────────────────────────────────┐              │
│  │              Host OS                   │              │
│  └───────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     Docker                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   App    │  │   App    │  │   App    │              │
│  ├──────────┤  ├──────────┤  ├──────────┤              │
│  │ 依赖/Bins │  │ 依赖/Bins │  │ 依赖/Bins │ ← 共享内核！   │
│  └──────────┘  └──────────┘  └──────────┘              │
│  ┌───────────────────────────────────────┐              │
│  │           Docker Engine               │              │
│  └───────────────────────────────────────┘              │
│  ┌───────────────────────────────────────┐              │
│  │              Host OS                   │              │
│  └───────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

| 特性 | 虚拟机 | Docker |
|------|--------|--------|
| **启动时间** | 分钟级 | 秒级 |
| **磁盘占用** | GB 级 | MB 级 |
| **性能** | 较低（虚拟化开销） | 接近原生 |
| **隔离性** | 完全隔离（更安全） | 进程级隔离 |
| **适用场景** | 需要完全隔离 | 应用部署、开发环境 |

---

## 2. 核心概念

### 2.1 镜像（Image）vs 容器（Container）

```
类比面向对象编程：

镜像 (Image) = 类 (Class)
  - 只读的模板
  - 包含运行应用所需的一切
  - 可以分享、存储

容器 (Container) = 实例 (Instance)
  - 镜像的运行实例
  - 可以启动、停止、删除
  - 相互隔离

┌─────────────────────────────────────────┐
│              Image: python:3.10         │
│                     │                   │
│         ┌──────────┼──────────┐         │
│         ▼          ▼          ▼         │
│    Container1  Container2  Container3   │
│    (运行中)    (运行中)    (已停止)     │
└─────────────────────────────────────────┘
```

```python
# 类比代码
class PythonImage:  # 镜像 = 类
    python_version = "3.10"
    packages = ["numpy", "pandas"]

container1 = PythonImage()  # 容器 = 实例
container2 = PythonImage()  # 可以创建多个容器
```

### 2.2 Dockerfile

```dockerfile
# Dockerfile = 构建镜像的配方

# 类比：
# Dockerfile 就像是一份详细的菜谱
# 告诉 Docker 如何一步步构建镜像
```

### 2.3 Docker Hub

```
Docker Hub = 镜像的 npm/PyPI

- 存储和分享镜像的仓库
- 官方镜像：python, ubuntu, nginx...
- 社区镜像：各种预配置环境
- 可以推送自己的镜像

常用镜像：
- python:3.10          # Python 官方镜像
- pytorch/pytorch      # PyTorch 官方镜像
- tensorflow/tensorflow # TensorFlow 官方镜像
- nvidia/cuda          # NVIDIA CUDA 镜像
```

### 2.4 概念关系图

```
Dockerfile ──build──> Image ──run──> Container
   (菜谱)              (菜)          (上桌的菜)
                        │
                        │ push/pull
                        ▼
                   Docker Hub
                   (菜谱仓库)
```

---

## 3. 安装与配置

### 3.1 安装 Docker

**macOS**:
```bash
# 方法 1：使用 Homebrew
brew install --cask docker

# 方法 2：下载 Docker Desktop
# https://www.docker.com/products/docker-desktop

# 启动 Docker Desktop 应用
```

**Ubuntu**:
```bash
# 更新包索引
sudo apt-get update

# 安装依赖
sudo apt-get install ca-certificates curl gnupg

# 添加 Docker 官方 GPG 密钥
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 设置仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 将当前用户添加到 docker 组（免 sudo）
sudo usermod -aG docker $USER
newgrp docker
```

**Windows**:
```
1. 下载 Docker Desktop: https://www.docker.com/products/docker-desktop
2. 启用 WSL 2（Windows Subsystem for Linux）
3. 安装并启动 Docker Desktop
```

### 3.2 验证安装

```bash
# 检查 Docker 版本
docker --version
# Docker version 24.0.0, build ...

# 检查 Docker 是否运行
docker info

# 运行测试容器
docker run hello-world
# 如果看到 "Hello from Docker!" 说明安装成功
```

### 3.3 配置镜像加速（国内用户）

```bash
# 编辑或创建 Docker 配置文件
# macOS: ~/.docker/daemon.json
# Linux: /etc/docker/daemon.json

{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}

# 重启 Docker
# macOS: 重启 Docker Desktop
# Linux: sudo systemctl restart docker
```

---

## 4. 基础命令

### 4.1 镜像操作

```bash
# 搜索镜像
docker search python
# NAME                 DESCRIPTION                                     STARS
# python               Python is an interpreted...                     9000+

# 拉取镜像
docker pull python:3.10
# 格式: docker pull <镜像名>:<标签>
# 不指定标签默认是 :latest

# 常用 Python 镜像
docker pull python:3.10           # 完整版（约 900MB）
docker pull python:3.10-slim      # 精简版（约 150MB）
docker pull python:3.10-alpine    # 超精简版（约 50MB，但兼容性差）

# 查看本地镜像
docker images
# REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
# python       3.10      abc123...      2 days ago    912MB

# 删除镜像
docker rmi python:3.10
# 或用镜像 ID
docker rmi abc123

# 清理未使用的镜像
docker image prune
# 清理所有未使用的镜像
docker image prune -a
```

### 4.2 容器操作

```bash
# 运行容器（最常用）
docker run python:3.10 python -c "print('Hello Docker!')"

# 运行容器参数详解
docker run \
  -it \                      # 交互模式 + 终端
  --name my-python \         # 容器名称
  -v $(pwd):/app \           # 挂载当前目录到容器的 /app
  -w /app \                  # 设置工作目录
  -p 8000:8000 \             # 端口映射 (主机:容器)
  -e MY_VAR=value \          # 环境变量
  python:3.10 \              # 使用的镜像
  bash                       # 运行的命令

# 常用参数说明
# -i: interactive，保持标准输入打开
# -t: tty，分配一个终端
# -d: detach，后台运行
# --rm: 退出后自动删除容器
# -v: volume，挂载目录
# -p: port，端口映射
# -e: environment，环境变量

# 实际示例
# 1. 进入 Python 交互环境
docker run -it python:3.10

# 2. 运行当前目录的脚本
docker run -it --rm -v $(pwd):/app -w /app python:3.10 python main.py

# 3. 启动 Jupyter（后台运行）
docker run -d -p 8888:8888 jupyter/scipy-notebook
```

### 4.3 容器管理

```bash
# 查看运行中的容器
docker ps
# CONTAINER ID   IMAGE         COMMAND   CREATED          STATUS          PORTS     NAMES
# abc123         python:3.10   "bash"    5 minutes ago    Up 5 minutes              my-python

# 查看所有容器（包括已停止的）
docker ps -a

# 进入运行中的容器
docker exec -it my-python bash
# exec: 在运行中的容器执行命令
# -it: 交互模式

# 停止容器
docker stop my-python

# 启动已停止的容器
docker start my-python

# 重启容器
docker restart my-python

# 删除容器
docker rm my-python
# 强制删除运行中的容器
docker rm -f my-python

# 查看容器日志
docker logs my-python
# 持续查看
docker logs -f my-python

# 查看容器详情
docker inspect my-python

# 清理所有已停止的容器
docker container prune
```

### 4.4 命令速查表

```bash
# 镜像相关
docker pull <image>        # 拉取镜像
docker images             # 列出镜像
docker rmi <image>        # 删除镜像
docker build -t <name> .  # 构建镜像

# 容器相关
docker run <image>        # 运行容器
docker ps                 # 列出运行中容器
docker ps -a              # 列出所有容器
docker stop <container>   # 停止容器
docker start <container>  # 启动容器
docker rm <container>     # 删除容器
docker exec -it <c> bash  # 进入容器

# 清理相关
docker system prune       # 清理所有未使用资源
docker system df          # 查看磁盘使用
```

---

## 5. Dockerfile 编写

### 5.1 基础语法

```dockerfile
# Dockerfile 示例

# FROM: 基础镜像（必须是第一条指令）
FROM python:3.10-slim

# LABEL: 元数据（可选）
LABEL maintainer="your@email.com"
LABEL version="1.0"

# ENV: 环境变量
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# WORKDIR: 设置工作目录
WORKDIR $APP_HOME

# COPY: 复制文件到镜像
# 格式: COPY <源路径> <目标路径>
COPY requirements.txt .

# RUN: 执行命令（构建时）
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# EXPOSE: 声明端口（文档作用，实际映射需要 -p）
EXPOSE 8000

# CMD: 容器启动时执行的命令
# 只能有一条 CMD，多条只有最后一条生效
CMD ["python", "main.py"]

# ENTRYPOINT: 容器入口点
# CMD 的参数会附加到 ENTRYPOINT 后面
# ENTRYPOINT ["python"]
# CMD ["main.py"]
```

### 5.2 常用指令详解

```dockerfile
# RUN vs CMD vs ENTRYPOINT

# RUN: 构建镜像时执行，结果会保存到镜像层
RUN apt-get update && apt-get install -y git
RUN pip install numpy pandas

# CMD: 容器启动时的默认命令，可以被 docker run 覆盖
CMD ["python", "app.py"]
# docker run myimage python other.py  # 会覆盖 CMD

# ENTRYPOINT: 容器入口点，不容易被覆盖
ENTRYPOINT ["python"]
CMD ["app.py"]
# docker run myimage other.py  # 实际执行 python other.py
```

```dockerfile
# COPY vs ADD

# COPY: 简单复制文件
COPY file.txt /app/
COPY . /app/

# ADD: 复制 + 额外功能（解压、URL下载）
ADD archive.tar.gz /app/      # 自动解压
ADD https://example.com/file /app/  # 下载（不推荐）

# 推荐：优先使用 COPY，更清晰
```

### 5.3 最佳实践

```dockerfile
# ✅ 好的 Dockerfile

FROM python:3.10-slim

# 1. 安装系统依赖（放在前面，缓存利用）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  # 清理缓存

WORKDIR /app

# 2. 先复制依赖文件，利用缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 再复制代码（代码变更频繁，放在后面）
COPY . .

# 4. 使用非 root 用户（安全）
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["python", "main.py"]
```

```dockerfile
# ❌ 不好的 Dockerfile

FROM python:3.10

# 每个 RUN 创建一层，层数太多
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y vim

# 代码放在依赖前面，每次代码变更都要重新安装依赖
COPY . /app
RUN pip install -r requirements.txt

# 没有清理缓存，镜像太大
```

### 5.4 AI 项目常用基础镜像

```dockerfile
# 1. 纯 Python
FROM python:3.10-slim

# 2. PyTorch（CPU）
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 3. TensorFlow（GPU）
FROM tensorflow/tensorflow:2.13.0-gpu

# 4. CUDA + cuDNN（自己装框架）
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 5. Jupyter
FROM jupyter/scipy-notebook
```

### 5.5 多阶段构建

```dockerfile
# 多阶段构建：减小最终镜像大小

# 阶段 1：构建
FROM python:3.10 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 阶段 2：运行
FROM python:3.10-slim

WORKDIR /app

# 只复制安装好的包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .
CMD ["python", "main.py"]

# 最终镜像只包含运行时需要的内容，不含构建工具
```

### 5.6 构建镜像

```bash
# 在 Dockerfile 所在目录执行
docker build -t my-app:1.0 .
# -t: 标签（名称:版本）
# .: 构建上下文（当前目录）

# 指定 Dockerfile
docker build -t my-app:1.0 -f Dockerfile.dev .

# 查看构建历史
docker history my-app:1.0

# 推送到 Docker Hub
docker login
docker tag my-app:1.0 username/my-app:1.0
docker push username/my-app:1.0
```

---

## 6. Docker Compose

### 6.1 什么是 Docker Compose

```yaml
# docker-compose.yml 用于定义和运行多容器应用

# 场景：你的 AI 应用需要
# - Python 应用服务器
# - PostgreSQL 数据库
# - Redis 缓存

# 不用 Compose：
docker run ... app
docker run ... postgres
docker run ... redis
# 还要配置网络、依赖关系...

# 用 Compose：
docker compose up
# 一条命令启动所有服务！
```

### 6.2 基础语法

```yaml
# docker-compose.yml

version: '3.8'  # Compose 文件版本

services:
  # 服务 1：Web 应用
  app:
    build: .                    # 使用当前目录的 Dockerfile 构建
    # 或者使用现成镜像
    # image: python:3.10
    ports:
      - "8000:8000"            # 端口映射
    volumes:
      - .:/app                  # 挂载当前目录
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    command: python main.py

  # 服务 2：数据库
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # 服务 3：缓存
  redis:
    image: redis:7-alpine

# 数据卷
volumes:
  postgres_data:
```

### 6.3 常用命令

```bash
# 启动所有服务
docker compose up
# 后台运行
docker compose up -d

# 停止所有服务
docker compose down
# 停止并删除数据卷
docker compose down -v

# 重新构建镜像
docker compose build
# 或
docker compose up --build

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs
docker compose logs app  # 某个服务的日志
docker compose logs -f   # 持续查看

# 进入某个服务的容器
docker compose exec app bash

# 运行一次性命令
docker compose run app python manage.py migrate

# 扩缩容
docker compose up -d --scale app=3  # 启动 3 个 app 实例
```

### 6.4 AI 项目示例

```yaml
# docker-compose.yml - AI 应用示例

version: '3.8'

services:
  # FastAPI 应用
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./models:/app/models  # 模型文件
    environment:
      - MODEL_PATH=/app/models/model.pt
      - LOG_LEVEL=INFO
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  # Jupyter Notebook
  jupyter:
    image: jupyter/scipy-notebook
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
    environment:
      - JUPYTER_ENABLE_LAB=yes

  # MLflow 追踪服务器
  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db

  # PostgreSQL
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
      - POSTGRES_DB=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## 7. GPU 支持

### 7.1 安装 NVIDIA Container Toolkit

```bash
# Ubuntu/Debian

# 添加 NVIDIA 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 重启 Docker
sudo systemctl restart docker
```

### 7.2 检查 GPU

```bash
# 检查主机 GPU
nvidia-smi

# 在容器中检查 GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
# 输出应该显示 GPU 信息
```

### 7.3 运行 GPU 容器

```bash
# 使用所有 GPU
docker run --gpus all pytorch/pytorch python -c "import torch; print(torch.cuda.is_available())"

# 使用特定 GPU
docker run --gpus '"device=0"' ...
docker run --gpus '"device=0,1"' ...

# 使用指定数量的 GPU
docker run --gpus 2 ...
```

### 7.4 GPU + Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  gpu-app:
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # 或指定数量: 1
              capabilities: [gpu]
    volumes:
      - .:/app
    command: python train.py
```

### 7.5 常用 GPU 镜像

```bash
# PyTorch + CUDA
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# TensorFlow + GPU
docker pull tensorflow/tensorflow:2.13.0-gpu

# NVIDIA CUDA 基础镜像
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
docker pull nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

# Hugging Face Transformers
docker pull huggingface/transformers-pytorch-gpu
```

---

## 8. 实战：容器化 Python 项目

### 8.1 项目结构

```
my-ml-project/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── models/
│   └── .gitkeep
├── data/
│   └── .gitkeep
└── tests/
    └── test_model.py
```

### 8.2 requirements.txt

```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
numpy==1.26.2
pandas==2.1.3
scikit-learn==1.3.2
torch==2.1.1
pydantic==2.5.2
python-multipart==0.0.6
```

### 8.3 Dockerfile

```dockerfile
# Dockerfile

# 基础镜像
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目代码
COPY src/ ./src/
COPY models/ ./models/

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.4 .dockerignore

```
# .dockerignore

# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.egg-info
.eggs
dist
build

# 虚拟环境
venv
.venv
env

# IDE
.idea
.vscode
*.swp

# 测试和文档
tests
docs
*.md
!README.md

# 数据和模型（如果太大或敏感）
data/*.csv
data/*.parquet
*.pt
*.pth
*.h5

# 其他
.env
.env.*
*.log
.DS_Store
```

### 8.5 docker-compose.yml

```yaml
# docker-compose.yml

version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: my-ml-app:latest
    container_name: ml-api
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src            # 开发时热重载
      - ./models:/app/models      # 模型文件
      - ./data:/app/data          # 数据文件
    environment:
      - MODEL_PATH=/app/models/model.pt
      - LOG_LEVEL=INFO
      - ENV=development
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # 可选：开发用 Jupyter
  jupyter:
    image: jupyter/scipy-notebook:latest
    container_name: ml-jupyter
    ports:
      - "8888:8888"
    volumes:
      - .:/home/jovyan/work
    environment:
      - JUPYTER_ENABLE_LAB=yes
    profiles:
      - dev  # 只在开发时启动

volumes:
  model_data:
```

### 8.6 主程序 src/main.py

```python
# src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI(title="ML API", version="1.0.0")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: float
    confidence: float

# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 预测端点
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        features = np.array(request.features)
        # 这里调用你的模型
        prediction = float(np.mean(features))  # 示例
        confidence = 0.95
        return PredictResponse(prediction=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "ML API is running"}
```

### 8.7 运行

```bash
# 构建镜像
docker compose build

# 启动服务
docker compose up -d

# 查看日志
docker compose logs -f app

# 测试 API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'

# 进入容器调试
docker compose exec app bash

# 停止服务
docker compose down
```

---

## 9. 常见问题排查

### 9.1 镜像拉取问题

```bash
# 问题：镜像拉取超时
# 解决：配置镜像加速（见 3.3 节）

# 问题：磁盘空间不足
docker system df          # 查看 Docker 磁盘使用
docker system prune -a    # 清理所有未使用资源

# 问题：权限不足
sudo usermod -aG docker $USER
newgrp docker
```

### 9.2 容器运行问题

```bash
# 问题：容器立即退出
# 原因：没有前台进程

# 查看退出日志
docker logs <container_id>

# 保持容器运行（调试用）
docker run -it --entrypoint /bin/bash myimage

# 问题：端口已被占用
# Error: bind: address already in use
lsof -i :8000             # 查看端口占用
docker ps                 # 查看运行中的容器
docker stop <container>   # 停止占用端口的容器
```

### 9.3 Volume 挂载问题

```bash
# 问题：挂载的文件没有权限
# 原因：容器内用户和主机用户 UID 不同

# 解决方法 1：在 Dockerfile 中指定 UID
RUN useradd -m -u 1000 appuser
USER appuser

# 解决方法 2：运行时指定用户
docker run -u $(id -u):$(id -g) ...

# 问题：Windows/Mac 路径问题
# 使用绝对路径或 $(pwd)
docker run -v "$(pwd):/app" ...
```

### 9.4 网络问题

```bash
# 问题：容器间无法通信
# 解决：使用同一网络

# 方法 1：Docker Compose（自动创建网络）

# 方法 2：创建自定义网络
docker network create mynetwork
docker run --network mynetwork --name app1 ...
docker run --network mynetwork --name app2 ...
# app2 中可以通过 http://app1:port 访问 app1

# 查看网络
docker network ls
docker network inspect mynetwork
```

### 9.5 GPU 问题

```bash
# 问题：容器内看不到 GPU
# 检查步骤：

# 1. 主机是否安装 NVIDIA 驱动
nvidia-smi

# 2. 是否安装 nvidia-container-toolkit
dpkg -l | grep nvidia-container

# 3. 运行时是否指定 --gpus
docker run --gpus all ...

# 4. CUDA 版本是否匹配
# 镜像的 CUDA 版本要 <= 主机驱动支持的版本
```

---

## 📚 命令速查表

```bash
# 镜像
docker pull <image>              # 拉取
docker build -t <name> .         # 构建
docker images                    # 列出
docker rmi <image>               # 删除

# 容器
docker run -it <image> bash      # 交互运行
docker run -d -p 8000:8000 ...   # 后台运行
docker ps                        # 列出运行中
docker logs <container>          # 查看日志
docker exec -it <c> bash         # 进入容器
docker stop/start/rm <c>         # 停止/启动/删除

# Compose
docker compose up -d             # 启动
docker compose down              # 停止
docker compose logs              # 日志
docker compose exec <svc> bash   # 进入服务

# 清理
docker system prune              # 清理未使用资源
```

---

## ➡️ 下一步

学完本节后，继续学习 [17-远程开发环境.md](./17-远程开发环境.md)

