# ============================================
# 多阶段构建 Dockerfile for Node.js
# ============================================

# ============================================
# 阶段 1: 依赖安装
# ============================================
# 使用完整 Node.js 镜像安装依赖
FROM node:18-alpine AS deps

# 设置工作目录
WORKDIR /app

# 复制 package 文件
# 单独复制这些文件可以利用 Docker 层缓存
# 只有 package.json 或 package-lock.json 变化时才重新安装依赖
COPY package.json package-lock.json* ./

# 安装所有依赖（包括 devDependencies，构建需要）
# 使用 npm ci 保证依赖版本与 lock 文件一致
RUN npm ci

# ============================================
# 阶段 2: 构建
# ============================================
FROM node:18-alpine AS builder

WORKDIR /app

# 从 deps 阶段复制 node_modules
COPY --from=deps /app/node_modules ./node_modules

# 复制源代码
COPY . .

# 构建 TypeScript
RUN npm run build

# 清理 devDependencies，只保留生产依赖
RUN npm prune --production

# ============================================
# 阶段 3: 生产运行
# ============================================
# 使用精简的 Alpine 镜像运行
FROM node:18-alpine AS runner

# 设置环境变量
ENV NODE_ENV=production
ENV PORT=3000

# 创建非 root 用户（安全最佳实践）
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nodejs

# 设置工作目录
WORKDIR /app

# 从 builder 阶段复制必要文件
# 只复制生产需要的文件，不包含源码和开发依赖
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./

# 切换到非 root 用户
USER nodejs

# 暴露端口（文档作用）
EXPOSE 3000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# 启动命令
CMD ["node", "dist/server.js"]

