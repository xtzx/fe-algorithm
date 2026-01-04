# P20: FastAPI 服务

> 构建生产级 API 服务

## 🎯 学完后能做

- 设计 RESTful API
- 实现认证与授权
- 构建可测试的服务

## 🚀 快速开始

```bash
# 进入项目目录
cd py-20-fastapi

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 运行服务
uvicorn api.main:app --reload

# 访问文档
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

## 📁 目录结构

```
py-20-fastapi/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-basics.md             # FastAPI 基础
│   ├── 02-pydantic.md           # pydantic 集成
│   ├── 03-dependencies.md       # 依赖注入
│   ├── 04-middleware.md         # 中间件
│   ├── 05-errors.md             # 错误处理
│   ├── 06-auth.md               # 认证与授权
│   ├── 07-testing.md            # 测试
│   ├── 08-exercises.md          # 练习题
│   └── 09-interview.md          # 面试题
├── src/api/
│   ├── __init__.py
│   ├── main.py                  # 应用入口
│   ├── config.py                # 配置管理
│   ├── routers/                 # 路由模块
│   │   ├── __init__.py
│   │   ├── users.py            # 用户路由
│   │   ├── items.py            # 商品路由
│   │   └── auth.py             # 认证路由
│   ├── schemas/                 # Pydantic 模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── item.py
│   │   └── auth.py
│   ├── services/                # 业务逻辑
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   └── item_service.py
│   ├── dependencies/            # 依赖注入
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── auth.py
│   ├── middleware/              # 中间件
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── trace.py
│   └── exceptions.py            # 异常处理
├── tests/
│   ├── conftest.py
│   ├── test_users.py
│   ├── test_items.py
│   └── test_auth.py
└── scripts/
    └── run_dev.sh
```

## 🆚 Python vs JavaScript 对比

| 概念 | Python (FastAPI) | JavaScript (Express) |
|------|------------------|----------------------|
| 路由 | `@app.get("/")` | `app.get("/", ...)` |
| 请求参数 | 类型注解 | `req.params` |
| 验证 | Pydantic | Joi / Zod |
| 依赖注入 | `Depends()` | 手动 / 装饰器 |
| 中间件 | `@app.middleware` | `app.use()` |
| 文档 | 自动生成 | Swagger 手动 |

## 🔧 核心功能

### 1. 基础路由

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

### 2. 请求验证

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### 3. 依赖注入

```python
from fastapi import Depends

async def get_db():
    db = Database()
    try:
        yield db
    finally:
        await db.close()

@app.get("/users/")
async def read_users(db = Depends(get_db)):
    return await db.get_users()
```

### 4. JWT 认证

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    return user
```

## 📚 学习路径

1. **FastAPI 基础** - 路由、参数、响应
2. **pydantic 集成** - 验证、序列化
3. **依赖注入** - Depends、数据库
4. **中间件** - CORS、日志、trace_id
5. **错误处理** - HTTPException、自定义处理器
6. **认证授权** - JWT、OAuth2
7. **测试** - TestClient、mock

## ✅ 功能清单

- [x] 路由与请求处理
- [x] 请求参数（path、query、body）
- [x] 响应模型
- [x] 状态码
- [x] pydantic 请求验证
- [x] 响应序列化
- [x] 文档自动生成
- [x] 依赖注入
- [x] CORS 中间件
- [x] 请求日志
- [x] trace_id
- [x] HTTPException
- [x] 自定义异常处理器
- [x] JWT 认证
- [x] 权限控制
- [x] TestClient 测试

