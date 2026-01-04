# ASGI 服务器

## 概述

ASGI（Asynchronous Server Gateway Interface）是 Python 异步 Web 应用的标准接口。

## 1. Uvicorn

Uvicorn 是一个轻量级的 ASGI 服务器。

### 1.1 安装

```bash
pip install uvicorn[standard]
```

### 1.2 基础使用

```bash
# 开发模式（自动重载）
uvicorn main:app --reload

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8000

# 指定 workers（仅限 Unix）
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 1.3 配置选项

```bash
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \           # Worker 进程数
    --log-level info \      # 日志级别
    --access-log \          # 启用访问日志
    --timeout-keep-alive 5  # Keep-alive 超时
```

### 1.4 代码中配置

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4,
        log_level="info",
        access_log=True,
    )
```

## 2. Gunicorn + Uvicorn Workers

生产环境推荐使用 Gunicorn 管理 Uvicorn workers。

### 2.1 安装

```bash
pip install gunicorn uvicorn[standard]
```

### 2.2 基础使用

```bash
gunicorn main:app \
    -w 4 \                            # Worker 数量
    -k uvicorn.workers.UvicornWorker \ # Worker 类型
    -b 0.0.0.0:8000                    # 绑定地址
```

### 2.3 完整配置

```bash
gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \              # 请求超时
    --graceful-timeout 30 \      # 优雅停机超时
    --keep-alive 5 \             # Keep-alive 超时
    --max-requests 1000 \        # Worker 最大请求数后重启
    --max-requests-jitter 50 \   # 随机抖动
    --preload \                  # 预加载应用
    --access-logfile - \         # 访问日志
    --error-logfile - \          # 错误日志
    --capture-output              # 捕获 stdout/stderr
```

### 2.4 配置文件

`gunicorn.conf.py`:

```python
# 服务器绑定
bind = "0.0.0.0:8000"

# Worker 配置
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# 超时配置
timeout = 120
graceful_timeout = 30
keepalive = 5

# Worker 重启策略
max_requests = 1000
max_requests_jitter = 50

# 预加载
preload_app = True

# 日志配置
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

# 安全配置
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
```

运行：

```bash
gunicorn main:app -c gunicorn.conf.py
```

## 3. Worker 数量计算

```python
# 公式: (2 * CPU核心数) + 1
import multiprocessing

workers = (2 * multiprocessing.cpu_count()) + 1
```

对于 I/O 密集型应用，可以适当增加。

## 4. 进程管理

### 4.1 Systemd

`/etc/systemd/system/myapp.service`:

```ini
[Unit]
Description=My FastAPI Application
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/myapp
Environment="PATH=/opt/myapp/.venv/bin"
ExecStart=/opt/myapp/.venv/bin/gunicorn main:app -c gunicorn.conf.py
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl enable myapp
sudo systemctl start myapp
sudo systemctl status myapp

# 重载配置
sudo systemctl daemon-reload
sudo systemctl reload myapp
```

### 4.2 Supervisor

`/etc/supervisor/conf.d/myapp.conf`:

```ini
[program:myapp]
command=/opt/myapp/.venv/bin/gunicorn main:app -c gunicorn.conf.py
directory=/opt/myapp
user=www-data
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/myapp/error.log
stdout_logfile=/var/log/myapp/access.log
```

## 5. Uvicorn vs Gunicorn

| 特性 | Uvicorn | Gunicorn + Uvicorn |
|------|---------|-------------------|
| 简单性 | 简单 | 需要配置 |
| 多进程 | 单进程或 --workers | 内置进程管理 |
| 进程重启 | 手动 | 自动（max_requests）|
| 信号处理 | 基础 | 完善 |
| 适用场景 | 开发/小项目 | 生产环境 |


