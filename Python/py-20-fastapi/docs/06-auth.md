# 认证与授权

## 概述

FastAPI 支持多种认证方式：

1. **JWT Bearer Token** - 最常用
2. **OAuth2** - 标准协议
3. **API Key** - 简单场景
4. **Session/Cookie** - 传统 Web

## 1. JWT 认证

### 1.1 安装依赖

```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

### 1.2 配置

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    secret_key: str = "your-secret-key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
```

### 1.3 密码处理

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return pwd_context.hash(password)
```

### 1.4 Token 生成与验证

```python
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt

def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """创建 JWT Token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)

    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm,
    )

def decode_token(token: str) -> dict:
    """解码 JWT Token"""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="无效的认证凭据",
        )
```

### 1.5 OAuth2 Scheme

```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """登录获取 Token"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=30),
    )

    return {"access_token": access_token, "token_type": "bearer"}
```

### 1.6 获取当前用户

```python
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """获取当前登录用户"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user(username)
    if user is None:
        raise credentials_exception

    return user

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

## 2. OAuth2 with Scopes

### 2.1 定义 Scopes

```python
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "users:read": "读取用户信息",
        "users:write": "修改用户信息",
        "items:read": "读取商品信息",
        "items:write": "修改商品信息",
        "admin": "管理员权限",
    },
)
```

### 2.2 在 Token 中包含 Scopes

```python
def create_access_token(data: dict, scopes: list[str]) -> str:
    to_encode = data.copy()
    to_encode["scopes"] = scopes
    # ... 其他代码
```

### 2.3 验证 Scopes

```python
from fastapi import Security
from fastapi.security import SecurityScopes

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
):
    # 解码 token
    payload = decode_token(token)
    username = payload.get("sub")
    token_scopes = payload.get("scopes", [])

    # 获取用户
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401)

    # 验证 scopes
    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=403,
                detail="权限不足",
            )

    return user

# 使用
@app.get("/users")
async def list_users(
    current_user: User = Security(get_current_user, scopes=["users:read"]),
):
    return get_all_users()

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Security(get_current_user, scopes=["admin"]),
):
    return delete_user_by_id(user_id)
```

## 3. API Key 认证

### 3.1 Header 方式

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != "valid-api-key":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/items")
async def list_items(api_key: str = Depends(get_api_key)):
    return []
```

### 3.2 Query 方式

```python
from fastapi.security import APIKeyQuery

api_key_query = APIKeyQuery(name="api_key")

async def get_api_key(api_key: str = Depends(api_key_query)):
    if api_key != "valid-api-key":
        raise HTTPException(status_code=403)
    return api_key
```

## 4. 权限控制

### 4.1 角色检查

```python
def require_role(allowed_roles: list[str]):
    """角色检查依赖"""
    async def role_checker(
        current_user: User = Depends(get_current_user),
    ):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="权限不足",
            )
        return current_user
    return role_checker

@app.get("/admin")
async def admin_panel(
    user: User = Depends(require_role(["admin"])),
):
    return {"message": "Welcome, admin!"}
```

### 4.2 资源所有者检查

```python
async def get_item_owner(
    item_id: int,
    current_user: User = Depends(get_current_user),
):
    """检查是否是商品所有者"""
    item = db.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404)

    if item.owner_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="只能操作自己的商品",
        )

    return item

@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item_data: ItemUpdate,
    item: Item = Depends(get_item_owner),
):
    return db.update_item(item_id, item_data)
```

### 4.3 组合权限

```python
async def require_admin_or_owner(
    item_id: int,
    current_user: User = Depends(get_current_user),
):
    """管理员或所有者"""
    # 管理员直接通过
    if "admin" in current_user.scopes:
        return current_user

    # 检查所有权
    item = db.get_item(item_id)
    if item and item.owner_id == current_user.id:
        return current_user

    raise HTTPException(status_code=403, detail="权限不足")
```

## 5. Refresh Token

```python
from pydantic import BaseModel

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

def create_token_pair(user: User) -> TokenPair:
    """创建访问令牌和刷新令牌"""
    access_token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=15),
    )

    refresh_token = create_access_token(
        data={"sub": user.username, "type": "refresh"},
        expires_delta=timedelta(days=7),
    )

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
    )

@app.post("/token/refresh")
async def refresh_token(refresh_token: str):
    """刷新访问令牌"""
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="无效的刷新令牌")

        username = payload.get("sub")
        user = get_user(username)

        # 创建新的访问令牌
        access_token = create_access_token(
            data={"sub": username, "type": "access"},
            expires_delta=timedelta(minutes=15),
        )

        return {"access_token": access_token, "token_type": "bearer"}
    except JWTError:
        raise HTTPException(status_code=401, detail="无效的刷新令牌")
```

## 6. 完整示例

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel

# 配置
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# 模型
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: str
    is_active: bool = True

# 工具函数
def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)

    user = fake_users_db.get(username)
    if not user:
        raise HTTPException(status_code=401)
    return User(**user)

# 路由
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="认证失败")

    access_token = create_access_token(data={"sub": user["username"]})
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

## Python vs JavaScript 对比

| 特性 | FastAPI | Express + Passport |
|------|---------|-------------------|
| JWT | `python-jose` | `jsonwebtoken` |
| 密码 | `passlib` | `bcrypt` |
| OAuth2 | 内置 | `passport-oauth2` |
| 依赖 | `Depends()` | 中间件 |
| Scopes | `Security()` | 自定义 |

