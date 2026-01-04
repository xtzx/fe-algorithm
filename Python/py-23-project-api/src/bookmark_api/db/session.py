"""
数据库会话管理
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bookmark_api.config import get_settings
from bookmark_api.db.models import Base

settings = get_settings()

# 创建引擎
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

# 创建会话工厂
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    数据库会话依赖

    用于 FastAPI 依赖注入
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """初始化数据库（创建表）"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


def drop_db():
    """删除所有表"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All database tables dropped")

