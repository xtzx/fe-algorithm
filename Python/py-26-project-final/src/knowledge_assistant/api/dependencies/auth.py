"""
认证依赖

提供:
- JWT Token 生成与验证
- 用户认证
- 权限控制
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from knowledge_assistant.api.schemas.auth import TokenData, UserResponse
from knowledge_assistant.config import get_settings

settings = get_settings()

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_prefix}/auth/token",
    auto_error=False,
)


# 简单的内存用户存储（生产环境应使用数据库）
_users_db: Dict[str, dict] = {}


def init_default_user():
    """初始化默认用户"""
    if "admin" not in _users_db:
        _users_db["admin"] = {
            "id": "user_admin",
            "username": "admin",
            "email": "admin@example.com",
            "hashed_password": get_password_hash("admin123"),
            "is_active": True,
            "scopes": ["user", "admin"],
            "created_at": datetime.now(timezone.utc),
        }


# 初始化
init_default_user()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    创建 JWT access token
    
    Args:
        data: 要编码的数据
        expires_delta: 过期时间增量
    
    Returns:
        JWT token 字符串
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """
    解码 JWT token
    
    Args:
        token: JWT token 字符串
    
    Returns:
        TokenData 对象
    
    Raises:
        HTTPException: token 无效时抛出
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
        scopes = payload.get("scopes", [])
        return TokenData(username=username, scopes=scopes)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    验证用户名和密码
    
    Args:
        username: 用户名
        password: 密码
    
    Returns:
        验证成功返回用户字典，失败返回 None
    """
    user = _users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_user(username: str, email: str, password: str) -> dict:
    """创建用户"""
    if username in _users_db:
        raise ValueError("用户名已存在")
    
    user_id = f"user_{hashlib.md5(username.encode()).hexdigest()[:8]}"
    user = {
        "id": user_id,
        "username": username,
        "email": email,
        "hashed_password": get_password_hash(password),
        "is_active": True,
        "scopes": ["user"],
        "created_at": datetime.now(timezone.utc),
    }
    _users_db[username] = user
    return user


def get_user(username: str) -> Optional[dict]:
    """获取用户"""
    return _users_db.get(username)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> UserResponse:
    """
    获取当前用户依赖
    
    从 JWT token 中解析用户信息
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要认证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = decode_token(token)
    
    if token_data.username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无法验证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        is_active=user["is_active"],
        scopes=user["scopes"],
        created_at=user["created_at"],
    )


async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user),
) -> UserResponse:
    """
    获取当前活跃用户
    
    在 get_current_user 基础上检查用户是否激活
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用",
        )
    return current_user


async def require_admin(
    current_user: UserResponse = Depends(get_current_active_user),
) -> UserResponse:
    """
    要求管理员权限
    """
    if "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )
    return current_user


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[UserResponse]:
    """
    可选用户认证
    
    如果提供了有效 token 则返回用户，否则返回 None
    """
    if token is None:
        return None
    
    try:
        return await get_current_user(token)
    except HTTPException:
        return None


