# 05. CI/CD Pipeline with GitHub Actions

> 从代码提交到自动部署的完整流水线

---

## 📑 目录

1. [CI/CD 基础](#cicd-基础)
2. [GitHub Actions 入门](#github-actions-入门)
3. [完整 Pipeline 示例](#完整-pipeline-示例)
4. [前端项目优化](#前端项目优化)
5. [常见错误 & 排查](#常见错误--排查)
6. [面试问答](#面试问答)

---

## CI/CD 基础

### 什么是 CI/CD

```
CI (Continuous Integration) 持续集成:
├── 代码提交后自动运行
├── Lint 检查
├── 单元测试
├── 构建验证
└── 目标：尽早发现问题

CD (Continuous Delivery/Deployment) 持续交付/部署:
├── CI 通过后自动执行
├── 构建产物
├── 推送到仓库/服务器
└── 目标：快速、可靠地发布
```

### 典型流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      CI/CD Pipeline                             │
│                                                                 │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐            │
│  │ 代码   │──►│  Lint  │──►│  Test  │──►│ Build  │            │
│  │ 提交   │   │ 检查   │   │ 测试   │   │ 构建   │            │
│  └────────┘   └────────┘   └────────┘   └────┬───┘            │
│                                              │                  │
│                        ┌─────────────────────┘                  │
│                        ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                    CD Pipeline                              ││
│  │                                                             ││
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐               ││
│  │  │构建Docker│──►│ 推送镜像 │──►│ 部署服务 │               ││
│  │  │  镜像    │   │ Registry │   │          │               ││
│  │  └──────────┘   └──────────┘   └──────────┘               ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## GitHub Actions 入门

### 基本概念

| 概念 | 说明 |
|------|------|
| **Workflow** | 工作流，由一个 YAML 文件定义 |
| **Event** | 触发工作流的事件（push, PR 等） |
| **Job** | 任务，包含多个步骤 |
| **Step** | 步骤，执行单个命令或 Action |
| **Action** | 可复用的操作单元 |
| **Runner** | 执行工作流的虚拟机 |

### 文件位置

```
.github/
└── workflows/
    ├── ci.yml
    ├── deploy.yml
    └── release.yml
```

### 基本语法

```yaml
name: CI                          # 工作流名称

on:                               # 触发条件
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:                             # 任务定义
  build:                          # 任务名称
    runs-on: ubuntu-latest        # 运行环境

    steps:                        # 步骤列表
      - name: Checkout            # 步骤名称
        uses: actions/checkout@v4 # 使用 Action

      - name: Run tests
        run: npm test             # 执行命令
```

---

## 完整 Pipeline 示例

### 前端项目 CI/CD

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '18'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ============================================
  # Job 1: 代码检查与测试
  # ============================================
  lint-and-test:
    name: Lint & Test
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'           # 缓存 npm 依赖

      - name: Install dependencies
        run: npm ci              # 使用 ci 而不是 install

      - name: Run ESLint
        run: npm run lint

      - name: Run Type Check
        run: npm run type-check

      - name: Run Tests
        run: npm run test -- --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        if: always()

  # ============================================
  # Job 2: 构建
  # ============================================
  build:
    name: Build
    runs-on: ubuntu-latest
    needs: lint-and-test         # 依赖 lint-and-test

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build
        env:
          # 注入构建时环境变量
          VITE_API_URL: ${{ vars.API_URL }}
          VITE_APP_VERSION: ${{ github.sha }}

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build-output
          path: dist/
          retention-days: 7

  # ============================================
  # Job 3: 构建 Docker 镜像
  # ============================================
  docker:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: build-output
          path: dist/

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ============================================
  # Job 4: 部署
  # ============================================
  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: docker
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    environment: production      # 需要手动批准

    steps:
      - name: Deploy to server
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            cd /opt/app
            docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            docker-compose up -d --force-recreate
            docker image prune -f
```

---

## 前端项目优化

### 1. 缓存 node_modules

```yaml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '18'
    cache: 'npm'        # 自动缓存 ~/.npm

# 或手动配置缓存
- name: Cache node_modules
  uses: actions/cache@v4
  with:
    path: node_modules
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

### 2. 多环境配置

```yaml
jobs:
  deploy:
    strategy:
      matrix:
        environment: [staging, production]

    environment: ${{ matrix.environment }}

    steps:
      - name: Build with env
        run: npm run build
        env:
          VITE_API_URL: ${{ vars.API_URL }}  # 从 environment 读取
```

### 3. 环境变量注入

```yaml
# Vite 项目
env:
  VITE_API_URL: https://api.example.com
  VITE_APP_VERSION: ${{ github.sha }}

# Next.js 项目
env:
  NEXT_PUBLIC_API_URL: https://api.example.com
```

### 4. 并行任务

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps: [...]

  test:
    runs-on: ubuntu-latest
    steps: [...]

  build:
    needs: [lint, test]  # 等待 lint 和 test 都完成
    runs-on: ubuntu-latest
    steps: [...]
```

---

## 常见错误 & 排查

### 1. 权限不足

```yaml
# 错误：permission denied

# 解决：添加 permissions
permissions:
  contents: read
  packages: write
```

### 2. 缓存失效

```yaml
# 问题：每次都重新安装依赖

# 确保 key 正确
key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}

# 检查：package-lock.json 是否提交
```

### 3. Secrets 未配置

```yaml
# 错误：secret not found

# 解决：在 GitHub 仓库 Settings > Secrets 中添加
# Secrets: SERVER_HOST, SERVER_USER, SERVER_SSH_KEY
```

### 4. 构建产物丢失

```yaml
# 问题：deploy job 找不到 build 产物

# 原因：不同 job 运行在不同机器
# 解决：使用 artifact 传递

# Job 1: 上传
- uses: actions/upload-artifact@v4
  with:
    name: build
    path: dist/

# Job 2: 下载
- uses: actions/download-artifact@v4
  with:
    name: build
    path: dist/
```

---

## 面试问答

### Q1: CI 和 CD 的区别是什么？

**答案**：

> **CI (持续集成)**：
> - 代码提交后自动触发
> - 运行 lint、测试、构建
> - 目标：尽早发现集成问题
>
> **CD (持续交付/部署)**：
> - CI 通过后自动执行
> - 构建产物、推送镜像、部署服务
> - 持续交付：需要手动批准发布
> - 持续部署：完全自动发布

### Q2: 如何优化 CI 构建速度？

**答案**：

> 几个关键策略：
>
> 1. **缓存依赖**：缓存 node_modules 或 npm cache
> 2. **并行任务**：lint 和 test 可以同时运行
> 3. **增量构建**：只构建变化的部分
> 4. **使用 npm ci**：比 npm install 更快
> 5. **精简 Docker 镜像**：减少推送时间

### Q3: GitHub Actions 中 secrets 和 variables 有什么区别？

**答案**：

> | 类型 | 存储位置 | 访问方式 | 适用场景 |
> |------|---------|---------|---------|
> | Secrets | 加密存储 | `${{ secrets.NAME }}` | 密码、API Key、SSH Key |
> | Variables | 明文存储 | `${{ vars.NAME }}` | 配置项、URL、版本号 |
>
> Secrets 在日志中会被自动遮蔽（显示为 ***）。

