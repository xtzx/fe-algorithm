"""
数据库依赖

演示如何使用依赖注入管理数据库连接
"""

from typing import Generator


class Database:
    """模拟数据库连接"""

    def __init__(self):
        self.connected = False

    def connect(self):
        """建立连接"""
        self.connected = True
        print("📦 Database connected")

    def disconnect(self):
        """断开连接"""
        self.connected = False
        print("📦 Database disconnected")

    def execute(self, query: str) -> list:
        """执行查询"""
        if not self.connected:
            raise RuntimeError("Database not connected")
        print(f"📦 Executing: {query}")
        return []


def get_db() -> Generator[Database, None, None]:
    """
    数据库依赖

    使用 yield 实现资源的自动清理

    Usage:
        @app.get("/items")
        async def get_items(db: Database = Depends(get_db)):
            return db.execute("SELECT * FROM items")
    """
    db = Database()
    db.connect()
    try:
        yield db
    finally:
        db.disconnect()


# 异步版本
class AsyncDatabase:
    """模拟异步数据库连接"""

    def __init__(self):
        self.connected = False

    async def connect(self):
        """建立连接"""
        self.connected = True
        print("📦 Async database connected")

    async def disconnect(self):
        """断开连接"""
        self.connected = False
        print("📦 Async database disconnected")

    async def execute(self, query: str) -> list:
        """执行查询"""
        if not self.connected:
            raise RuntimeError("Database not connected")
        print(f"📦 Async executing: {query}")
        return []


async def get_async_db():
    """
    异步数据库依赖

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncDatabase = Depends(get_async_db)):
            return await db.execute("SELECT * FROM items")
    """
    db = AsyncDatabase()
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()

