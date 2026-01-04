"""
认证依赖

提供:
- JWT Token 生成与验证
- 用户认证
- 权限控制
"""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext

from api.config import get_settings
from api.schemas.auth import TokenData
from api.schemas.user import User
from api.services.user_service import UserService

settings = get_settings()

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_prefix}/auth/token",
    scopes={
        "user": "普通用户权限",
        "admin": "管理员权限",
    },
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
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
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm,
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
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        username: str | None = payload.get("sub")
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


def authenticate_user(
    user_service: UserService,
    username: str,
    password: str,
) -> User | None:
    """
    验证用户名和密码

    Args:
        user_service: 用户服务
        username: 用户名
        password: 密码

    Returns:
        验证成功返回 User 对象，失败返回 None
    """
    user = user_service.get_user_in_db(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return User(**user.model_dump(exclude={"hashed_password"}))


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
    user_service: UserService = Depends(),
) -> User:
    """
    获取当前用户依赖

    从 JWT token 中解析用户信息

    Usage:
        @app.get("/users/me")
        async def read_users_me(current_user: User = Depends(get_current_user)):
            return current_user
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": authenticate_value},
    )

    try:
        token_data = decode_token(token)
    except HTTPException:
        raise credentials_exception

    if token_data.username is None:
        raise credentials_exception

    user = user_service.get_user_by_username(token_data.username)
    if user is None:
        raise credentials_exception

    # 检查权限范围
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="没有足够的权限",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
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
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    要求管理员权限

    Usage:
        @app.delete("/users/{user_id}")
        async def delete_user(
            user_id: int,
            admin: User = Depends(require_admin)
        ):
            ...
    """
    if "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )
    return current_user

