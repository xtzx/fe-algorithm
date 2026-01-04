"""
JWT 令牌处理
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from jose import JWTError, jwt

from bookmark_api.config import get_settings

settings = get_settings()


def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    创建访问令牌

    Args:
        data: 要编码的数据
        expires_delta: 过期时间增量

    Returns:
        JWT 令牌
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )

    to_encode.update({"exp": expire, "type": "access"})

    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    创建刷新令牌

    Args:
        data: 要编码的数据
        expires_delta: 过期时间增量

    Returns:
        JWT 令牌
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.jwt_refresh_token_expire_days
        )

    to_encode.update({"exp": expire, "type": "refresh"})

    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> Optional[dict[str, Any]]:
    """
    解码 JWT 令牌

    Args:
        token: JWT 令牌

    Returns:
        解码后的数据，如果无效返回 None
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None


def verify_token_type(token: str, expected_type: str) -> Optional[dict[str, Any]]:
    """
    验证令牌类型

    Args:
        token: JWT 令牌
        expected_type: 期望的类型 ("access" 或 "refresh")

    Returns:
        解码后的数据，如果无效返回 None
    """
    payload = decode_token(token)
    if payload and payload.get("type") == expected_type:
        return payload
    return None

