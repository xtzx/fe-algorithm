# 🚀 DevOps for 前端/Node 工程师

> Docker 容器化 + Nginx 配置 + CI/CD Pipeline 实战指南

## 📚 项目简介

本项目面向 **7-8 年经验的资深前端工程师**，帮助你：

1. 掌握 **Docker 容器化** 的核心概念与实践
2. 理解 **Nginx** 反向代理、负载均衡、静态资源服务配置
3. 设计并落地 **CI/CD Pipeline**
4. 准备 DevOps 相关的 **面试问题**

---

## 📁 项目结构

```
devops-for-fe/
├── README.md                           # 本文件
├── docs/
│   ├── 01-docker-basics.md             # Docker 核心概念
│   ├── 02-dockerfile-node-example.md   # Dockerfile 多阶段构建
│   ├── 03-docker-compose-node-redis-nginx.md  # Docker Compose 实战
│   ├── 04-nginx-core-config.md         # Nginx 核心配置
│   ├── 05-ci-cd-pipeline-with-github-actions.md  # CI/CD Pipeline
│   └── 06-end-to-end-flow-and-interview.md  # 端到端流程 & 面试
├── examples/
│   ├── node-app/                       # Node.js 示例应用
│   │   ├── package.json
│   │   ├── src/server.ts
│   │   └── Dockerfile
│   ├── nginx/                          # Nginx 配置示例
│   │   ├── nginx.conf
│   │   └── site.conf
│   ├── docker-compose/                 # Docker Compose 示例
│   │   └── docker-compose.yml
│   └── ci/                             # CI/CD 配置示例
│       └── github-actions-node.yml
└── scripts/                            # 运维脚本
    ├── build-and-run.sh
    └── deploy-with-docker.sh
```

---

## 🎯 学习路线

```
Step 1: Docker 基础
├── 镜像、容器、仓库概念
├── Dockerfile 编写
└── 多阶段构建实践
        │
        ▼
Step 2: Docker Compose
├── 多服务编排
├── 网络与数据卷
└── 开发环境搭建
        │
        ▼
Step 3: Nginx 配置
├── 反向代理
├── 负载均衡
├── 静态资源 & 缓存
└── HTTPS 配置
        │
        ▼
Step 4: CI/CD Pipeline
├── GitHub Actions 配置
├── 自动化测试 & 构建
└── Docker 镜像发布
        │
        ▼
Step 5: 端到端部署
├── 完整部署流程
└── 面试问题准备
```

---

## 🔥 核心技能点

### Docker

| 技能点 | 重要性 | 说明 |
|--------|:------:|------|
| Dockerfile 编写 | ⭐⭐⭐⭐⭐ | 多阶段构建、镜像优化 |
| Docker Compose | ⭐⭐⭐⭐⭐ | 本地开发环境编排 |
| 镜像优化 | ⭐⭐⭐⭐ | 减小体积、加速构建 |
| 网络配置 | ⭐⭐⭐ | 容器间通信 |

### Nginx

| 技能点 | 重要性 | 说明 |
|--------|:------:|------|
| 反向代理 | ⭐⭐⭐⭐⭐ | proxy_pass、upstream |
| 静态资源服务 | ⭐⭐⭐⭐⭐ | SPA 部署、try_files |
| 负载均衡 | ⭐⭐⭐⭐ | 多节点、策略选择 |
| 缓存配置 | ⭐⭐⭐⭐ | expires、cache-control |
| HTTPS | ⭐⭐⭐ | 证书配置 |

### CI/CD

| 技能点 | 重要性 | 说明 |
|--------|:------:|------|
| GitHub Actions | ⭐⭐⭐⭐⭐ | Pipeline 配置 |
| 自动化测试 | ⭐⭐⭐⭐⭐ | lint、test |
| Docker 构建发布 | ⭐⭐⭐⭐ | 镜像构建、推送 |
| 环境变量管理 | ⭐⭐⭐⭐ | 多环境配置 |

---

## 🚀 快速开始

### 本地运行示例

```bash
# 1. 进入 docker-compose 目录
cd examples/docker-compose

# 2. 启动所有服务
docker-compose up -d

# 3. 访问应用
# - 前端: http://localhost
# - API: http://localhost/api
```

### 构建 Node 应用镜像

```bash
# 进入 node-app 目录
cd examples/node-app

# 构建镜像
docker build -t my-node-app .

# 运行容器
docker run -p 3000:3000 my-node-app
```

---

## 📖 推荐阅读顺序

1. `docs/01-docker-basics.md` - Docker 核心概念
2. `docs/02-dockerfile-node-example.md` - Dockerfile 实战
3. `docs/03-docker-compose-node-redis-nginx.md` - 多服务编排
4. `docs/04-nginx-core-config.md` - Nginx 配置
5. `docs/05-ci-cd-pipeline-with-github-actions.md` - CI/CD
6. `docs/06-end-to-end-flow-and-interview.md` - 综合 & 面试

---

## 🔗 参考资源

- [Docker 官方文档](https://docs.docker.com/)
- [Nginx 官方文档](https://nginx.org/en/docs/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [Docker Hub](https://hub.docker.com/)

