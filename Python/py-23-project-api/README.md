# P23: 书签管理 API

> 综合项目 4 - 完整的 API 服务

## 🎯 项目目标

开发一个「书签管理 API」，功能包括：
- 用户认证（JWT）
- 书签 CRUD
- 分类与标签
- 搜索与分页
- 数据导入导出
- 完整的测试与部署配置

## 🚀 快速开始

### 开发环境

```bash
# 克隆并进入项目
cd py-23-project-api

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 初始化数据库
alembic upgrade head

# 启动开发服务器
uvicorn bookmark_api.main:app --reload

# API 文档
# http://localhost:8000/docs
```

### Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f app

# 停止服务
docker-compose down
```

## 📁 目录结构

```
py-23-project-api/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
├── src/bookmark_api/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py          # 认证路由
│   │   ├── users.py         # 用户路由
│   │   ├── bookmarks.py     # 书签路由
│   │   ├── categories.py    # 分类路由
│   │   └── tags.py          # 标签路由
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py          # 用户 Schema
│   │   ├── bookmark.py      # 书签 Schema
│   │   ├── category.py      # 分类 Schema
│   │   ├── tag.py           # 标签 Schema
│   │   └── common.py        # 通用 Schema
│   ├── services/
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   ├── bookmark_service.py
│   │   └── export_service.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py        # SQLAlchemy 模型
│   │   ├── session.py       # 数据库会话
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── user_repo.py
│   │       └── bookmark_repo.py
│   ├── cache/
│   │   ├── __init__.py
│   │   └── client.py        # Redis 缓存
│   └── auth/
│       ├── __init__.py
│       ├── jwt.py           # JWT 处理
│       ├── password.py      # 密码哈希
│       └── dependencies.py  # 认证依赖
├── tests/
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_bookmarks.py
│   └── test_users.py
└── scripts/
    ├── run_dev.sh
    ├── run_prod.sh
    └── test.sh
```

## 🔧 API 端点

### 认证 `/api/v1/auth`

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/register` | 用户注册 |
| POST | `/login` | 用户登录 |
| POST | `/refresh` | 刷新令牌 |
| POST | `/logout` | 用户登出 |

### 用户 `/api/v1/users`

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/me` | 获取当前用户 |
| PUT | `/me` | 更新当前用户 |
| DELETE | `/me` | 删除账户 |

### 书签 `/api/v1/bookmarks`

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 获取书签列表（分页） |
| POST | `/` | 创建书签 |
| GET | `/{id}` | 获取书签详情 |
| PUT | `/{id}` | 更新书签 |
| DELETE | `/{id}` | 删除书签 |
| GET | `/search` | 搜索书签 |
| POST | `/import` | 导入书签 |
| GET | `/export` | 导出书签 |

### 分类 `/api/v1/categories`

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 获取分类列表 |
| POST | `/` | 创建分类 |
| PUT | `/{id}` | 更新分类 |
| DELETE | `/{id}` | 删除分类 |

### 标签 `/api/v1/tags`

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 获取标签列表 |
| POST | `/` | 创建标签 |
| DELETE | `/{id}` | 删除标签 |

## 📝 API 示例

### 注册

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "email": "user@example.com", "password": "password123"}'
```

### 登录

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password123"}'
```

### 创建书签

```bash
curl -X POST http://localhost:8000/api/v1/bookmarks \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "title": "Example", "category_id": 1}'
```

## ✅ 功能清单

- [x] RESTful API 设计
- [x] API 版本控制 (v1)
- [x] 分页与排序
- [x] 用户注册/登录
- [x] JWT 令牌认证
- [x] 刷新令牌
- [x] SQLAlchemy 模型
- [x] Repository 模式
- [x] 数据库迁移 (Alembic)
- [x] Redis 缓存
- [x] 缓存失效策略
- [x] Docker 配置
- [x] 健康检查
- [x] 结构化日志
- [x] 单元测试

