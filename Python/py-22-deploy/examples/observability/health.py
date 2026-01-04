"""
健康检查

演示:
- 存活检查 (Liveness)
- 就绪检查 (Readiness)
- 依赖检查
- Kubernetes 集成
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


@dataclass
class CheckResult:
    """检查结果"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    duration_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class HealthReport:
    """健康报告"""
    status: HealthStatus
    timestamp: str
    version: str
    checks: List[CheckResult] = field(default_factory=list)


# ==================== 健康检查器 ====================


class HealthChecker:
    """
    健康检查管理器
    
    Usage:
        checker = HealthChecker(version="1.0.0")
        
        # 注册检查
        checker.add_check("database", check_database)
        checker.add_check("redis", check_redis)
        
        # 执行检查
        report = await checker.check_health()
    """
    
    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self._checks: Dict[str, Callable] = {}
    
    def add_check(self, name: str, check_func: Callable):
        """添加检查函数"""
        self._checks[name] = check_func
    
    def remove_check(self, name: str):
        """移除检查函数"""
        self._checks.pop(name, None)
    
    async def _run_check(self, name: str, check_func: Callable) -> CheckResult:
        """运行单个检查"""
        import time
        
        start = time.perf_counter()
        try:
            # 支持同步和异步检查函数
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration = (time.perf_counter() - start) * 1000
            
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                return CheckResult(name=name, status=status, duration_ms=duration)
            
            if isinstance(result, dict):
                status = HealthStatus.HEALTHY if result.get("ok", True) else HealthStatus.UNHEALTHY
                return CheckResult(
                    name=name,
                    status=status,
                    message=result.get("message"),
                    duration_ms=duration,
                    details=result.get("details"),
                )
            
            return CheckResult(name=name, status=HealthStatus.HEALTHY, duration_ms=duration)
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration,
            )
    
    async def check_health(self) -> HealthReport:
        """执行所有健康检查"""
        # 并发执行所有检查
        tasks = [
            self._run_check(name, func)
            for name, func in self._checks.items()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 确定总体状态
        if all(r.status == HealthStatus.HEALTHY for r in results):
            overall_status = HealthStatus.HEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        return HealthReport(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version=self.version,
            checks=list(results),
        )
    
    async def check_liveness(self) -> HealthReport:
        """存活检查（只检查应用是否在运行）"""
        return HealthReport(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.utcnow().isoformat(),
            version=self.version,
        )
    
    async def check_readiness(self) -> HealthReport:
        """就绪检查（检查所有依赖）"""
        return await self.check_health()


# ==================== 常见检查函数 ====================


async def check_database(connection_string: str = None) -> dict:
    """数据库连接检查"""
    try:
        # 这里应该实际执行数据库连接检查
        # 示例：await database.execute("SELECT 1")
        return {"ok": True, "message": "Database is reachable"}
    except Exception as e:
        return {"ok": False, "message": f"Database error: {e}"}


async def check_redis(url: str = None) -> dict:
    """Redis 连接检查"""
    try:
        # 这里应该实际执行 Redis 连接检查
        # 示例：await redis.ping()
        return {"ok": True, "message": "Redis is reachable"}
    except Exception as e:
        return {"ok": False, "message": f"Redis error: {e}"}


async def check_external_api(url: str) -> dict:
    """外部 API 检查"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 200:
                return {"ok": True, "message": f"API responded with {response.status_code}"}
            return {
                "ok": False,
                "message": f"API returned status {response.status_code}",
            }
    except Exception as e:
        return {"ok": False, "message": f"API error: {e}"}


def check_disk_space(threshold_percent: float = 90.0) -> dict:
    """磁盘空间检查"""
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    used_percent = (used / total) * 100
    
    if used_percent > threshold_percent:
        return {
            "ok": False,
            "message": f"Disk usage {used_percent:.1f}% exceeds threshold {threshold_percent}%",
            "details": {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
            },
        }
    
    return {
        "ok": True,
        "message": f"Disk usage {used_percent:.1f}%",
        "details": {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
        },
    }


def check_memory(threshold_percent: float = 90.0) -> dict:
    """内存检查"""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        
        if used_percent > threshold_percent:
            return {
                "ok": False,
                "message": f"Memory usage {used_percent}% exceeds threshold {threshold_percent}%",
            }
        
        return {
            "ok": True,
            "message": f"Memory usage {used_percent}%",
            "details": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
            },
        }
    except ImportError:
        return {"ok": True, "message": "psutil not installed, skipping memory check"}


# ==================== FastAPI 集成 ====================


def create_health_router(checker: HealthChecker):
    """创建健康检查路由"""
    from fastapi import APIRouter, HTTPException
    
    router = APIRouter(tags=["Health"])
    
    @router.get("/health")
    async def health():
        """存活检查"""
        report = await checker.check_liveness()
        return {
            "status": report.status.value,
            "timestamp": report.timestamp,
            "version": report.version,
        }
    
    @router.get("/health/ready")
    async def ready():
        """就绪检查"""
        report = await checker.check_readiness()
        
        if report.status == HealthStatus.UNHEALTHY:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": report.status.value,
                    "checks": [
                        {
                            "name": c.name,
                            "status": c.status.value,
                            "message": c.message,
                        }
                        for c in report.checks
                    ],
                },
            )
        
        return {
            "status": report.status.value,
            "timestamp": report.timestamp,
            "version": report.version,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "duration_ms": c.duration_ms,
                }
                for c in report.checks
            ],
        }
    
    return router


# ==================== 使用示例 ====================


async def demo_health():
    """演示健康检查"""
    
    checker = HealthChecker(version="1.0.0")
    
    # 添加检查
    checker.add_check("disk", check_disk_space)
    checker.add_check("memory", check_memory)
    checker.add_check("database", lambda: {"ok": True, "message": "Mock DB OK"})
    
    # 执行检查
    report = await checker.check_health()
    
    print(f"Status: {report.status.value}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Version: {report.version}")
    print("Checks:")
    for check in report.checks:
        print(f"  - {check.name}: {check.status.value} ({check.duration_ms:.2f}ms)")
        if check.message:
            print(f"    Message: {check.message}")


if __name__ == "__main__":
    asyncio.run(demo_health())


