# ============================================
# 多阶段构建 Dockerfile
# ============================================
# 优势：
# 1. 镜像更小 - 不包含构建工具
# 2. 更安全 - 减少攻击面
# 3. 更快部署 - 镜像体积小
# ============================================

# ==================== 构建阶段 ====================
FROM python:3.11-slim as builder

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装依赖到用户目录（便于复制到运行阶段）
RUN pip install --user -r requirements.txt


# ==================== 运行阶段 ====================
FROM python:3.11-slim as runtime

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # 应用配置
    APP_ENV=production \
    APP_PORT=8000 \
    # Python 路径
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY . .

# 创建非 root 用户（安全最佳实践）
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


