"""
数据库会话管理

提供:
- 同步会话
- 异步会话
- 会话依赖（用于 FastAPI）
"""

from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from storage_lab.config import get_settings
from storage_lab.db.models import Base

settings = get_settings()


# ==================== 同步引擎和会话 ====================

engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,  # 连接健康检查
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    同步数据库会话依赖

    用于 FastAPI 的依赖注入

    Usage:
        @app.get("/users")
        def list_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== 异步引擎和会话 ====================

async_engine = create_async_engine(
    settings.async_database_url,
    echo=settings.database_echo,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    异步数据库会话依赖

    用于 FastAPI 的异步依赖注入

    Usage:
        @app.get("/users")
        async def list_users(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ==================== 数据库初始化 ====================


def init_db():
    """
    初始化数据库（创建所有表）

    注意：生产环境应使用 Alembic 迁移
    """
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


async def init_async_db():
    """
    异步初始化数据库
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Async database tables created")


def drop_db():
    """
    删除所有表（仅用于测试）
    """
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All database tables dropped")


