# 03. Docker Compose 多服务编排

> 用 Docker Compose 编排 Node.js + Nginx + Redis 服务

---

## 📑 目录

1. [Docker Compose 简介](#docker-compose-简介)
2. [核心配置字段](#核心配置字段)
3. [实战：完整的服务编排](#实战完整的服务编排)
4. [常用命令](#常用命令)
5. [常见错误 & 排查](#常见错误--排查)
6. [面试问答](#面试问答)

---

## Docker Compose 简介

Docker Compose 是用于定义和运行多容器 Docker 应用的工具。通过一个 YAML 文件描述所有服务，一条命令启动整个应用。

### 使用场景

```
开发环境:
├── 前端容器 (Nginx 静态服务)
├── 后端容器 (Node.js API)
├── 数据库容器 (Redis / MySQL)
└── 一键启动整个环境

测试环境:
├── 与生产环境一致的配置
└── CI/CD 中自动化测试
```

---

## 核心配置字段

### 基本结构

```yaml
version: '3.8'  # Compose 文件版本

services:       # 服务定义
  web:
    ...
  api:
    ...
  redis:
    ...

networks:       # 网络配置（可选）
  ...

volumes:        # 数据卷配置（可选）
  ...
```

### services - 服务定义

```yaml
services:
  api:
    # ============================================
    # 镜像相关
    # ============================================
    image: node:18-alpine          # 使用现有镜像
    # 或
    build:                         # 从 Dockerfile 构建
      context: ./node-app          # 构建上下文路径
      dockerfile: Dockerfile       # Dockerfile 文件名

    # ============================================
    # 容器配置
    # ============================================
    container_name: my-api         # 容器名称
    restart: unless-stopped        # 重启策略
    # restart 选项:
    #   - no: 不自动重启
    #   - always: 总是重启
    #   - on-failure: 失败时重启
    #   - unless-stopped: 除非手动停止

    # ============================================
    # 端口映射
    # ============================================
    ports:
      - "3000:3000"                # 主机端口:容器端口
      - "9229:9229"                # 调试端口

    # ============================================
    # 环境变量
    # ============================================
    environment:
      - NODE_ENV=development
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    # 或从文件读取
    env_file:
      - .env

    # ============================================
    # 数据卷
    # ============================================
    volumes:
      - ./node-app:/app            # 绑定挂载（开发用）
      - /app/node_modules          # 匿名卷（保护 node_modules）
      - data-volume:/app/data      # 命名卷

    # ============================================
    # 依赖关系
    # ============================================
    depends_on:
      - redis                      # 先启动 redis
      - db

    # ============================================
    # 网络
    # ============================================
    networks:
      - app-network

    # ============================================
    # 健康检查
    # ============================================
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

### networks - 网络

```yaml
networks:
  app-network:
    driver: bridge                 # 默认桥接网络

  # 使用已存在的网络
  external-network:
    external: true
```

### volumes - 数据卷

```yaml
volumes:
  # 命名卷（Docker 管理）
  data-volume:

  # 指定驱动
  db-data:
    driver: local
```

---

## 实战：完整的服务编排

### 架构图

```
                    ┌─────────────────┐
                    │     Nginx       │ :80
                    │  (反向代理)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌───────────┐  ┌───────────┐  ┌───────────┐
       │  静态资源  │  │  API 请求  │  │  WebSocket│
       │  /static  │  │  /api/*   │  │  /ws      │
       └───────────┘  └─────┬─────┘  └───────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Node.js     │ :3000
                    │   (API 服务)   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     Redis     │ :6379
                    │   (缓存/会话)  │
                    └───────────────┘
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  # ============================================
  # Nginx - 反向代理 + 静态资源服务
  # ============================================
  nginx:
    image: nginx:alpine
    container_name: nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      # Nginx 配置
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/site.conf:/etc/nginx/conf.d/default.conf:ro
      # 静态资源
      - ./frontend/dist:/usr/share/nginx/html:ro
      # SSL 证书（如需要）
      # - ./certs:/etc/nginx/certs:ro
    depends_on:
      - api
    networks:
      - app-network

  # ============================================
  # Node.js API 服务
  # ============================================
  api:
    build:
      context: ./node-app
      dockerfile: Dockerfile
    container_name: api
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - PORT=3000
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    env_file:
      - .env
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ============================================
  # Redis - 缓存/会话存储
  # ============================================
  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

# ============================================
# 网络
# ============================================
networks:
  app-network:
    driver: bridge

# ============================================
# 数据卷
# ============================================
volumes:
  redis-data:
```

### 开发环境配置

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  api:
    build:
      context: ./node-app
      target: development          # 多阶段构建的开发阶段
    volumes:
      - ./node-app:/app            # 绑定挂载，支持热重载
      - /app/node_modules          # 保护容器内的 node_modules
    environment:
      - NODE_ENV=development
    command: npm run dev           # 覆盖默认命令

  nginx:
    volumes:
      - ./frontend/src:/usr/share/nginx/html  # 开发时的源码目录
```

使用方式：
```bash
# 开发环境
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# 生产环境
docker-compose up -d
```

---

## 常用命令

```bash
# 启动所有服务
docker-compose up

# 后台启动
docker-compose up -d

# 启动特定服务
docker-compose up api redis

# 停止服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v

# 重新构建镜像
docker-compose up --build

# 查看服务日志
docker-compose logs -f api

# 查看服务状态
docker-compose ps

# 进入容器
docker-compose exec api sh

# 重启服务
docker-compose restart api

# 扩容服务
docker-compose up -d --scale api=3
```

---

## 常见错误 & 排查

### 1. 服务启动顺序问题

```yaml
# 问题：depends_on 只保证启动顺序，不保证服务就绪

# 解决：使用 healthcheck + condition
depends_on:
  redis:
    condition: service_healthy
```

### 2. 容器间无法通信

```bash
# 问题：api 连不上 redis

# 排查
# 1. 确认在同一网络
docker network inspect app-network

# 2. 使用服务名作为主机名
REDIS_HOST=redis  # 不是 localhost 或 IP
```

### 3. 端口冲突

```bash
# 错误：bind: address already in use

# 解决：修改主机端口
ports:
  - "3001:3000"  # 主机 3001 映射到容器 3000
```

### 4. 数据卷权限问题

```bash
# 问题：容器内无法写入挂载的目录

# 解决
# 1. 确保主机目录有正确权限
chmod 755 ./data

# 2. 或在容器内以 root 运行（不推荐生产环境）
```

---

## 面试问答

### Q1: Docker Compose 的 depends_on 能保证服务完全就绪吗？

**答案**：

> 不能。`depends_on` 只保证 **容器启动顺序**，不保证服务就绪。
>
> 例如，Redis 容器启动了，但 Redis 服务可能还在初始化。
>
> **解决方案**：
> 1. 使用 `healthcheck` + `condition: service_healthy`
> 2. 应用层实现重试逻辑
> 3. 使用 wait-for 脚本

### Q2: 开发环境和生产环境的 docker-compose 配置有什么区别？

**答案**：

> | 维度 | 开发环境 | 生产环境 |
> |------|---------|---------|
> | 代码挂载 | 绑定挂载 (volumes) | 镜像内置 |
> | 热重载 | 开启 | 关闭 |
> | 调试端口 | 暴露 | 不暴露 |
> | 日志级别 | debug | info/warn |
> | 资源限制 | 无 | 有 |
>
> 通常使用 `docker-compose.override.yml` 或多文件组合实现差异化配置。

### Q3: volumes 的绑定挂载和命名卷有什么区别？

**答案**：

> | 类型 | 语法 | 特点 |
> |------|------|------|
> | 绑定挂载 | `./host/path:/container/path` | 直接映射主机目录，适合开发 |
> | 命名卷 | `volume-name:/container/path` | Docker 管理，适合生产数据持久化 |
>
> **最佳实践**：
> - 开发：绑定挂载（代码同步）
> - 生产：命名卷（数据持久化）

