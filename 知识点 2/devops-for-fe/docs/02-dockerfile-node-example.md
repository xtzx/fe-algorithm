# 02. Dockerfile 多阶段构建实战

> 为 Node.js 项目编写高效的 Dockerfile

---

## 📑 目录

1. [Dockerfile 基础](#dockerfile-基础)
2. [常用指令详解](#常用指令详解)
3. [多阶段构建](#多阶段构建)
4. [最佳实践](#最佳实践)
5. [常见错误 & 排查](#常见错误--排查)
6. [面试问答](#面试问答)

---

## Dockerfile 基础

Dockerfile 是一个文本文件，包含构建 Docker 镜像的所有指令。

### 基本结构

```dockerfile
# 基础镜像
FROM node:18-alpine

# 工作目录
WORKDIR /app

# 复制文件
COPY package*.json ./

# 安装依赖
RUN npm install

# 复制源码
COPY . .

# 构建
RUN npm run build

# 暴露端口
EXPOSE 3000

# 启动命令
CMD ["npm", "start"]
```

---

## 常用指令详解

### FROM - 基础镜像

```dockerfile
# 语法
FROM <image>:<tag>

# 示例
FROM node:18-alpine      # 推荐：体积小
FROM node:18-slim        # 中等体积
FROM node:18             # 完整版，体积大

# 多阶段构建可以有多个 FROM
FROM node:18 AS builder
FROM node:18-alpine AS runner
```

### WORKDIR - 工作目录

```dockerfile
# 设置后续指令的工作目录
WORKDIR /app

# 如果目录不存在会自动创建
# 之后的 RUN、CMD、COPY 等都在此目录执行
```

### COPY vs ADD

```dockerfile
# COPY: 简单复制（推荐）
COPY package.json ./
COPY src/ ./src/

# ADD: 额外支持 URL 和解压 tar
ADD https://example.com/file.tar.gz /app/
ADD archive.tar.gz /app/  # 自动解压

# 最佳实践：优先使用 COPY，只在需要解压时用 ADD
```

### RUN - 执行命令

```dockerfile
# Shell 形式
RUN npm install

# Exec 形式（推荐用于多参数命令）
RUN ["npm", "install", "--production"]

# 合并多个命令减少层数
RUN npm install && \
    npm run build && \
    npm cache clean --force
```

### ENV - 环境变量

```dockerfile
# 设置环境变量
ENV NODE_ENV=production
ENV PORT=3000

# 多个变量
ENV NODE_ENV=production \
    PORT=3000
```

### EXPOSE - 声明端口

```dockerfile
# 仅做文档说明，不会真正发布端口
EXPOSE 3000

# 运行时需要 -p 参数才能映射端口
# docker run -p 3000:3000 my-app
```

### CMD vs ENTRYPOINT

```dockerfile
# CMD: 可被 docker run 参数覆盖
CMD ["npm", "start"]
# docker run my-app npm run dev  → 执行 npm run dev

# ENTRYPOINT: 固定入口点
ENTRYPOINT ["npm"]
CMD ["start"]
# docker run my-app run dev  → 执行 npm run dev

# 最佳实践
# - 单一用途容器用 ENTRYPOINT
# - 通用容器用 CMD
```

### VOLUME - 数据卷

```dockerfile
# 声明挂载点
VOLUME ["/app/data", "/app/logs"]

# 运行时挂载
# docker run -v /host/data:/app/data my-app
```

---

## 多阶段构建

多阶段构建是优化镜像大小的关键技术。

### 问题：传统单阶段构建

```dockerfile
# ❌ 不推荐：最终镜像包含构建工具和开发依赖
FROM node:18

WORKDIR /app
COPY . .
RUN npm install
RUN npm run build

CMD ["npm", "start"]

# 结果：镜像可能超过 1GB
```

### 解决：多阶段构建

```dockerfile
# ============================================
# 阶段 1: 构建阶段 (Builder)
# ============================================
FROM node:18 AS builder

WORKDIR /app

# 先复制 package.json，利用缓存
COPY package*.json ./

# 安装所有依赖（包括 devDependencies）
RUN npm ci

# 复制源码
COPY . .

# 构建
RUN npm run build

# ============================================
# 阶段 2: 生产阶段 (Runner)
# ============================================
FROM node:18-alpine AS runner

WORKDIR /app

# 设置生产环境
ENV NODE_ENV=production

# 只复制生产需要的文件
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./

# 只安装生产依赖
RUN npm ci --only=production && \
    npm cache clean --force

# 创建非 root 用户
RUN addgroup -g 1001 nodejs && \
    adduser -S -u 1001 -G nodejs nodejs

USER nodejs

EXPOSE 3000

CMD ["node", "dist/server.js"]
```

### 多阶段构建的优势

```
构建阶段 (node:18):
├── 包含完整的构建工具
├── 安装 devDependencies
├── 编译 TypeScript
└── 产出: dist/ 目录

          │
          ▼ 只复制需要的文件

生产阶段 (node:18-alpine):
├── 精简的基础镜像
├── 只有生产依赖
├── 只有编译后的代码
└── 最终镜像: ~100MB
```

---

## 最佳实践

### 1. 利用构建缓存

```dockerfile
# ✅ 好：先复制 package.json，依赖不变时使用缓存
COPY package*.json ./
RUN npm ci
COPY . .

# ❌ 差：每次源码变化都要重新安装依赖
COPY . .
RUN npm ci
```

### 2. 使用 .dockerignore

```dockerignore
# .dockerignore
node_modules
dist
.git
.env
*.log
.DS_Store
coverage
.nyc_output
```

### 3. 使用非 root 用户

```dockerfile
# 创建用户
RUN addgroup -g 1001 nodejs && \
    adduser -S -u 1001 -G nodejs nodejs

# 切换用户
USER nodejs
```

### 4. 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1
```

### 5. 使用 npm ci 而不是 npm install

```dockerfile
# npm ci: 精确安装 package-lock.json 中的版本，更快更可靠
RUN npm ci --only=production
```

---

## 常见错误 & 排查

### 1. 构建缓存失效

```bash
# 问题：每次构建都重新安装依赖

# 原因：COPY . . 在 RUN npm install 之前
# 任何文件变化都会使缓存失效

# 解决：先复制 package.json
COPY package*.json ./
RUN npm ci
COPY . .
```

### 2. 镜像过大

```bash
# 排查：查看各层大小
docker history my-app

# 解决方案
# 1. 使用 alpine 基础镜像
# 2. 多阶段构建
# 3. 清理缓存
RUN npm ci && npm cache clean --force
```

### 3. 构建失败：找不到文件

```bash
# 问题：COPY 的文件不存在

# 排查
# 1. 检查 .dockerignore 是否排除了该文件
# 2. 检查路径是否正确
# 3. 确认文件存在于构建上下文中

# 注意：Docker 只能访问构建上下文内的文件
docker build -t my-app .  # . 是构建上下文
```

### 4. 运行时端口无法访问

```bash
# 问题：容器运行了但访问不了

# 检查
# 1. EXPOSE 只是文档，需要 -p 映射
docker run -p 3000:3000 my-app

# 2. 应用是否监听 0.0.0.0
# Node.js 默认可能只监听 127.0.0.1
app.listen(3000, '0.0.0.0');
```

---

## 面试问答

### Q1: 什么是多阶段构建？有什么好处？

**答案**：

> 多阶段构建是指在一个 Dockerfile 中使用多个 `FROM` 指令，每个 `FROM` 开始一个新的构建阶段。
>
> **好处**：
> 1. **减小镜像体积**：最终镜像只包含运行需要的文件
> 2. **安全**：不暴露构建工具和源码
> 3. **简化流程**：无需维护多个 Dockerfile
>
> **示例**：
> - 构建阶段：使用完整的 Node.js 镜像编译 TypeScript
> - 生产阶段：使用 alpine 镜像，只复制编译后的 JS 文件
> - 结果：镜像从 1GB 减小到 100MB

### Q2: COPY 和 ADD 的区别？

**答案**：

> | 指令 | 功能 |
> |------|------|
> | COPY | 简单复制文件或目录 |
> | ADD | 复制 + 支持 URL 下载 + 自动解压 tar |
>
> **最佳实践**：优先使用 COPY，只在需要解压 tar 文件时使用 ADD。
>
> 原因：COPY 语义更明确，行为更可预测。

### Q3: CMD 和 ENTRYPOINT 的区别？

**答案**：

> | 指令 | 行为 |
> |------|------|
> | CMD | 默认命令，可被 `docker run` 参数覆盖 |
> | ENTRYPOINT | 固定入口，`docker run` 参数会作为追加参数 |
>
> **常见组合**：
> ```dockerfile
> ENTRYPOINT ["npm"]
> CMD ["start"]
> ```
> - `docker run my-app` → 执行 `npm start`
> - `docker run my-app run dev` → 执行 `npm run dev`

