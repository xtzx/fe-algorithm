# 04. Nginx 核心配置

> 反向代理、负载均衡、静态资源服务、缓存

---

## 📑 目录

1. [Nginx 基础](#nginx-基础)
2. [反向代理](#反向代理)
3. [负载均衡](#负载均衡)
4. [静态资源服务](#静态资源服务)
5. [缓存配置](#缓存配置)
6. [HTTPS 配置](#https-配置)
7. [常见错误 & 排查](#常见错误--排查)
8. [面试问答](#面试问答)

---

## Nginx 基础

### 配置文件结构

```nginx
# 主配置文件: /etc/nginx/nginx.conf

# 全局配置
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

# 事件配置
events {
    worker_connections 1024;
}

# HTTP 配置
http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    keepalive_timeout 65;

    # 包含其他配置文件
    include /etc/nginx/conf.d/*.conf;
}
```

### 配置层级

```
http {
    # 全局 HTTP 配置

    server {
        # 虚拟主机配置

        location / {
            # 路径匹配配置
        }
    }
}
```

---

## 反向代理

### 基本代理

```nginx
server {
    listen 80;
    server_name example.com;

    # 代理所有请求到后端
    location / {
        proxy_pass http://localhost:3000;
    }
}
```

### 完整代理配置

```nginx
server {
    listen 80;
    server_name example.com;

    location /api/ {
        # 代理目标
        proxy_pass http://localhost:3000;

        # ============================================
        # 请求头设置
        # ============================================
        # 传递原始 Host
        proxy_set_header Host $host;

        # 传递真实客户端 IP
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # 传递协议（HTTP/HTTPS）
        proxy_set_header X-Forwarded-Proto $scheme;

        # ============================================
        # 超时设置
        # ============================================
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # ============================================
        # 缓冲设置
        # ============================================
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
}
```

### 常见代理头部

| 头部 | 作用 |
|------|------|
| `X-Real-IP` | 客户端真实 IP |
| `X-Forwarded-For` | 经过的代理链 |
| `X-Forwarded-Proto` | 原始请求协议 |
| `X-Forwarded-Host` | 原始请求 Host |

---

## 负载均衡

### upstream 配置

```nginx
# 定义上游服务器组
upstream backend {
    # 服务器列表
    server 192.168.1.10:3000 weight=3;  # 权重 3
    server 192.168.1.11:3000 weight=2;  # 权重 2
    server 192.168.1.12:3000;           # 默认权重 1

    # 备用服务器（其他全挂时启用）
    server 192.168.1.13:3000 backup;

    # 标记为不可用
    # server 192.168.1.14:3000 down;
}

server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
    }
}
```

### 负载均衡策略

```nginx
upstream backend {
    # ============================================
    # 1. 轮询 (Round Robin) - 默认
    # ============================================
    # 按顺序分配请求
    server 192.168.1.10:3000;
    server 192.168.1.11:3000;

    # ============================================
    # 2. 加权轮询 (Weighted Round Robin)
    # ============================================
    # 按权重分配
    server 192.168.1.10:3000 weight=3;
    server 192.168.1.11:3000 weight=1;

    # ============================================
    # 3. IP 哈希 (ip_hash)
    # ============================================
    # 同一 IP 始终访问同一服务器（会话保持）
    ip_hash;
    server 192.168.1.10:3000;
    server 192.168.1.11:3000;

    # ============================================
    # 4. 最少连接 (least_conn)
    # ============================================
    # 分配给当前连接数最少的服务器
    least_conn;
    server 192.168.1.10:3000;
    server 192.168.1.11:3000;

    # ============================================
    # 5. 哈希 (hash)
    # ============================================
    # 自定义哈希键
    hash $request_uri consistent;
    server 192.168.1.10:3000;
    server 192.168.1.11:3000;
}
```

### 健康检查

```nginx
upstream backend {
    server 192.168.1.10:3000 max_fails=3 fail_timeout=30s;
    server 192.168.1.11:3000 max_fails=3 fail_timeout=30s;

    # max_fails: 失败次数阈值
    # fail_timeout: 失败后暂停时间
}
```

---

## 静态资源服务

### 基本配置

```nginx
server {
    listen 80;
    server_name example.com;

    # 静态资源根目录
    root /usr/share/nginx/html;
    index index.html;

    # 静态资源请求
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### SPA 应用配置

```nginx
server {
    listen 80;
    server_name example.com;

    root /usr/share/nginx/html;
    index index.html;

    # ============================================
    # SPA 路由处理 (关键！)
    # ============================================
    location / {
        # 尝试顺序：
        # 1. 请求的文件 $uri
        # 2. 请求的目录 $uri/
        # 3. 回退到 index.html (SPA 路由)
        try_files $uri $uri/ /index.html;
    }

    # ============================================
    # 静态资源 (带哈希的文件可以长期缓存)
    # ============================================
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # ============================================
    # API 代理
    # ============================================
    location /api/ {
        proxy_pass http://backend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### try_files 详解

```nginx
try_files $uri $uri/ /index.html;

# 请求 /about
# 1. 尝试 /about 文件 → 不存在
# 2. 尝试 /about/ 目录 → 不存在
# 3. 返回 /index.html → SPA 处理路由
```

---

## 缓存配置

### 静态资源缓存

```nginx
server {
    listen 80;
    server_name example.com;
    root /usr/share/nginx/html;

    # ============================================
    # HTML - 不缓存或短期缓存
    # ============================================
    location ~* \.html$ {
        expires -1;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }

    # ============================================
    # 带哈希的静态资源 - 长期缓存
    # ============================================
    # 例如: main.abc123.js, style.def456.css
    location ~* \.[a-f0-9]{8,}\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # ============================================
    # 普通静态资源 - 中期缓存
    # ============================================
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 7d;
        add_header Cache-Control "public";
    }
}
```

### 缓存策略对比

| 资源类型 | 策略 | 说明 |
|----------|------|------|
| HTML | 不缓存 | 确保获取最新版本 |
| 带哈希的 JS/CSS | 1年 | 内容变化则哈希变化 |
| 图片/字体 | 7天-1月 | 按需调整 |
| API 响应 | 不缓存 | 动态数据 |

### ETag 配置

```nginx
# 启用 ETag（默认开启）
etag on;

# 启用 Last-Modified（默认开启）
if_modified_since exact;
```

---

## HTTPS 配置

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    # ============================================
    # SSL 证书
    # ============================================
    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    # ============================================
    # SSL 配置优化
    # ============================================
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000" always;

    # ... 其他配置
}

# HTTP 重定向到 HTTPS
server {
    listen 80;
    server_name example.com;
    return 301 https://$server_name$request_uri;
}
```

---

## 常见错误 & 排查

### 1. 502 Bad Gateway

```bash
# 原因：后端服务不可达

# 排查
# 1. 检查后端服务是否运行
curl http://localhost:3000

# 2. 检查 Nginx 错误日志
tail -f /var/log/nginx/error.log

# 3. 检查防火墙/网络
```

### 2. 404 Not Found

```bash
# 原因：文件不存在或路径错误

# 排查
# 1. 检查 root 目录是否正确
# 2. 检查文件权限
ls -la /usr/share/nginx/html/

# 3. SPA 应用检查 try_files 配置
```

### 3. SPA 刷新 404

```nginx
# 问题：SPA 直接访问 /about 返回 404

# 解决：添加 try_files
location / {
    try_files $uri $uri/ /index.html;
}
```

### 4. 配置语法错误

```bash
# 测试配置
nginx -t

# 重新加载配置
nginx -s reload
```

---

## 面试问答

### Q1: 正向代理和反向代理的区别？

**答案**：

> | 类型 | 代理对象 | 使用者 | 典型场景 |
> |------|---------|--------|---------|
> | 正向代理 | 客户端 | 客户端知道代理 | VPN、翻墙 |
> | 反向代理 | 服务端 | 客户端不知道代理 | 负载均衡、CDN |
>
> **反向代理**：客户端访问代理服务器，代理服务器将请求转发给后端服务器。客户端不知道真正的服务器是谁。

### Q2: Nginx 的负载均衡策略有哪些？

**答案**：

> 1. **轮询 (Round Robin)**：默认策略，按顺序分配
> 2. **加权轮询**：按权重分配，权重高的服务器处理更多请求
> 3. **IP 哈希 (ip_hash)**：同一 IP 始终访问同一服务器，用于会话保持
> 4. **最少连接 (least_conn)**：分配给当前连接数最少的服务器
> 5. **哈希 (hash)**：自定义哈希键（如 URL）

### Q3: 如何配置 SPA 应用的 Nginx？

**答案**：

> 核心是 `try_files` 指令：
>
> ```nginx
> location / {
>     try_files $uri $uri/ /index.html;
> }
> ```
>
> **原理**：
> 1. 先尝试请求的文件（静态资源）
> 2. 再尝试目录
> 3. 都不存在则返回 index.html，由前端路由处理
>
> **还需要注意**：
> - API 请求需要单独配置代理
> - 静态资源需要配置缓存

