"""
状态管理

支持断点续跑的状态持久化
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
from typing import Any


class TaskStatus(Enum):
    """任务状态"""

    PENDING = "pending"  # 待执行
    RUNNING = "running"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    SKIPPED = "skipped"  # 跳过


@dataclass
class TaskState:
    """单个任务状态"""

    index: int
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        return cls(
            index=data["index"],
            status=TaskStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            error=data.get("error"),
            attempts=data.get("attempts", 0),
        )


@dataclass
class BatchState:
    """批处理状态"""

    batch_id: str
    created_at: datetime
    updated_at: datetime
    total_tasks: int
    tasks: dict[int, TaskState] = field(default_factory=dict)

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)

    @property
    def running_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

    @property
    def skipped_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)

    @property
    def progress(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        done = self.completed_count + self.failed_count + self.skipped_count
        return done / self.total_tasks * 100

    @property
    def is_complete(self) -> bool:
        return self.pending_count == 0 and self.running_count == 0

    def get_summary(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "total": self.total_tasks,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "pending": self.pending_count,
            "skipped": self.skipped_count,
            "progress": f"{self.progress:.1f}%",
        }


class StateManager:
    """状态管理器"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._state: BatchState | None = None

    @property
    def state(self) -> BatchState | None:
        return self._state

    def init_state(self, batch_id: str, total_tasks: int) -> BatchState:
        """初始化新的批处理状态"""
        now = datetime.now()
        self._state = BatchState(
            batch_id=batch_id,
            created_at=now,
            updated_at=now,
            total_tasks=total_tasks,
            tasks={i: TaskState(index=i) for i in range(total_tasks)},
        )
        self.save()
        return self._state

    def load(self) -> BatchState | None:
        """加载状态"""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)

            self._state = BatchState(
                batch_id=data["batch_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                total_tasks=data["total_tasks"],
                tasks={
                    int(k): TaskState.from_dict(v) for k, v in data["tasks"].items()
                },
            )
            return self._state

        except (json.JSONDecodeError, KeyError) as e:
            print(f"警告：状态文件损坏: {e}")
            return None

    def save(self) -> None:
        """保存状态（原子写入）"""
        if self._state is None:
            return

        self._state.updated_at = datetime.now()

        data = {
            "batch_id": self._state.batch_id,
            "created_at": self._state.created_at.isoformat(),
            "updated_at": self._state.updated_at.isoformat(),
            "total_tasks": self._state.total_tasks,
            "tasks": {str(k): v.to_dict() for k, v in self._state.tasks.items()},
        }

        # 原子写入：先写临时文件，再重命名
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.rename(self.state_file)

    def mark_started(self, index: int) -> None:
        """标记任务开始"""
        if self._state and index in self._state.tasks:
            task = self._state.tasks[index]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.attempts += 1
            self.save()

    def mark_completed(self, index: int) -> None:
        """标记任务完成"""
        if self._state and index in self._state.tasks:
            task = self._state.tasks[index]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.error = None
            self.save()

    def mark_failed(self, index: int, error: str) -> None:
        """标记任务失败"""
        if self._state and index in self._state.tasks:
            task = self._state.tasks[index]
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error
            self.save()

    def mark_skipped(self, index: int, reason: str = "") -> None:
        """标记任务跳过"""
        if self._state and index in self._state.tasks:
            task = self._state.tasks[index]
            task.status = TaskStatus.SKIPPED
            task.completed_at = datetime.now()
            task.error = reason or "Skipped"
            self.save()

    def get_pending_indices(self) -> list[int]:
        """获取待执行任务索引"""
        if self._state is None:
            return []
        return sorted(
            i
            for i, t in self._state.tasks.items()
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        )

    def get_failed_indices(self) -> list[int]:
        """获取失败任务索引（用于重试）"""
        if self._state is None:
            return []
        return sorted(
            i for i, t in self._state.tasks.items() if t.status == TaskStatus.FAILED
        )

    def reset_failed(self) -> int:
        """重置失败任务为待执行状态"""
        if self._state is None:
            return 0

        count = 0
        for task in self._state.tasks.values():
            if task.status == TaskStatus.FAILED:
                task.status = TaskStatus.PENDING
                task.error = None
                count += 1

        if count > 0:
            self.save()

        return count

    def clear(self) -> None:
        """清除状态"""
        self._state = None
        if self.state_file.exists():
            self.state_file.unlink()


@dataclass
class RollbackEntry:
    """回滚日志条目"""

    index: int
    operation_type: str
    source: str
    target: str | None
    executed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operation_type": self.operation_type,
            "source": self.source,
            "target": self.target,
            "executed_at": self.executed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RollbackEntry":
        return cls(
            index=data["index"],
            operation_type=data["operation_type"],
            source=data["source"],
            target=data.get("target"),
            executed_at=datetime.fromisoformat(data["executed_at"]),
        )


class RollbackLog:
    """回滚日志"""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.entries: list[RollbackEntry] = []

    def load(self) -> None:
        """加载回滚日志"""
        if not self.log_file.exists():
            return

        try:
            with open(self.log_file, encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [RollbackEntry.from_dict(e) for e in data]
        except (json.JSONDecodeError, KeyError):
            self.entries = []

    def save(self) -> None:
        """保存回滚日志"""
        data = [e.to_dict() for e in self.entries]
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record(self, entry: RollbackEntry) -> None:
        """记录已执行的操作"""
        self.entries.append(entry)
        self.save()

    def get_rollback_entries(self) -> list[RollbackEntry]:
        """获取回滚条目（逆序）"""
        return list(reversed(self.entries))

    def remove_last(self) -> RollbackEntry | None:
        """移除最后一条记录"""
        if self.entries:
            entry = self.entries.pop()
            self.save()
            return entry
        return None

    def clear(self) -> None:
        """清空日志"""
        self.entries = []
        if self.log_file.exists():
            self.log_file.unlink()

