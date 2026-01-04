"""
测试状态管理模块
"""

import pytest
from pathlib import Path
from datetime import datetime

from file_automation.state import (
    StateManager,
    BatchState,
    TaskState,
    TaskStatus,
    RollbackLog,
    RollbackEntry,
)


class TestTaskState:
    """测试任务状态"""

    def test_to_dict(self):
        """测试序列化"""
        state = TaskState(
            index=0,
            status=TaskStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 0, 1),
            attempts=1,
        )
        data = state.to_dict()

        assert data["index"] == 0
        assert data["status"] == "completed"
        assert data["attempts"] == 1

    def test_from_dict(self):
        """测试反序列化"""
        data = {
            "index": 1,
            "status": "failed",
            "started_at": "2024-01-01T12:00:00",
            "completed_at": "2024-01-01T12:00:01",
            "error": "Some error",
            "attempts": 3,
        }
        state = TaskState.from_dict(data)

        assert state.index == 1
        assert state.status == TaskStatus.FAILED
        assert state.error == "Some error"
        assert state.attempts == 3


class TestBatchState:
    """测试批处理状态"""

    def test_counts(self):
        """测试状态计数"""
        state = BatchState(
            batch_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            total_tasks=5,
            tasks={
                0: TaskState(0, TaskStatus.COMPLETED),
                1: TaskState(1, TaskStatus.COMPLETED),
                2: TaskState(2, TaskStatus.FAILED),
                3: TaskState(3, TaskStatus.PENDING),
                4: TaskState(4, TaskStatus.PENDING),
            },
        )

        assert state.completed_count == 2
        assert state.failed_count == 1
        assert state.pending_count == 2
        assert state.progress == 60.0  # 3/5 done

    def test_is_complete(self):
        """测试是否完成"""
        state = BatchState(
            batch_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            total_tasks=2,
            tasks={
                0: TaskState(0, TaskStatus.COMPLETED),
                1: TaskState(1, TaskStatus.COMPLETED),
            },
        )

        assert state.is_complete is True


class TestStateManager:
    """测试状态管理器"""

    def test_init_state(self, state_file: Path):
        """测试初始化状态"""
        manager = StateManager(state_file)
        state = manager.init_state("batch_001", 5)

        assert state.batch_id == "batch_001"
        assert state.total_tasks == 5
        assert len(state.tasks) == 5
        assert state_file.exists()

    def test_save_and_load(self, state_file: Path):
        """测试保存和加载"""
        manager = StateManager(state_file)
        manager.init_state("batch_001", 3)
        manager.mark_completed(0)
        manager.mark_failed(1, "Error")

        # 重新加载
        manager2 = StateManager(state_file)
        state = manager2.load()

        assert state is not None
        assert state.batch_id == "batch_001"
        assert state.tasks[0].status == TaskStatus.COMPLETED
        assert state.tasks[1].status == TaskStatus.FAILED
        assert state.tasks[1].error == "Error"

    def test_mark_started(self, state_file: Path):
        """测试标记开始"""
        manager = StateManager(state_file)
        manager.init_state("batch", 1)
        manager.mark_started(0)

        assert manager.state is not None
        assert manager.state.tasks[0].status == TaskStatus.RUNNING
        assert manager.state.tasks[0].attempts == 1

    def test_mark_completed(self, state_file: Path):
        """测试标记完成"""
        manager = StateManager(state_file)
        manager.init_state("batch", 1)
        manager.mark_completed(0)

        assert manager.state is not None
        assert manager.state.tasks[0].status == TaskStatus.COMPLETED

    def test_mark_failed(self, state_file: Path):
        """测试标记失败"""
        manager = StateManager(state_file)
        manager.init_state("batch", 1)
        manager.mark_failed(0, "Test error")

        assert manager.state is not None
        assert manager.state.tasks[0].status == TaskStatus.FAILED
        assert manager.state.tasks[0].error == "Test error"

    def test_get_pending_indices(self, state_file: Path):
        """测试获取待执行索引"""
        manager = StateManager(state_file)
        manager.init_state("batch", 5)
        manager.mark_completed(0)
        manager.mark_completed(1)
        manager.mark_failed(2, "Error")

        pending = manager.get_pending_indices()
        assert pending == [3, 4]

    def test_reset_failed(self, state_file: Path):
        """测试重置失败任务"""
        manager = StateManager(state_file)
        manager.init_state("batch", 3)
        manager.mark_failed(0, "Error 1")
        manager.mark_failed(1, "Error 2")
        manager.mark_completed(2)

        count = manager.reset_failed()

        assert count == 2
        assert manager.state is not None
        assert manager.state.tasks[0].status == TaskStatus.PENDING
        assert manager.state.tasks[1].status == TaskStatus.PENDING

    def test_clear(self, state_file: Path):
        """测试清除状态"""
        manager = StateManager(state_file)
        manager.init_state("batch", 1)
        manager.clear()

        assert manager.state is None
        assert not state_file.exists()


class TestRollbackLog:
    """测试回滚日志"""

    def test_record_and_load(self, rollback_file: Path):
        """测试记录和加载"""
        log = RollbackLog(rollback_file)
        log.record(
            RollbackEntry(
                index=0,
                operation_type="rename",
                source="/path/to/old",
                target="/path/to/new",
                executed_at=datetime.now(),
            )
        )

        # 重新加载
        log2 = RollbackLog(rollback_file)
        log2.load()

        assert len(log2.entries) == 1
        assert log2.entries[0].operation_type == "rename"

    def test_get_rollback_entries(self, rollback_file: Path):
        """测试获取回滚条目（逆序）"""
        log = RollbackLog(rollback_file)
        for i in range(3):
            log.record(
                RollbackEntry(
                    index=i,
                    operation_type="rename",
                    source=f"/src/{i}",
                    target=f"/dst/{i}",
                    executed_at=datetime.now(),
                )
            )

        entries = log.get_rollback_entries()

        assert len(entries) == 3
        assert entries[0].index == 2  # 最后一个变成第一个
        assert entries[2].index == 0

    def test_remove_last(self, rollback_file: Path):
        """测试移除最后一条"""
        log = RollbackLog(rollback_file)
        log.record(
            RollbackEntry(
                index=0,
                operation_type="rename",
                source="/a",
                target="/b",
                executed_at=datetime.now(),
            )
        )
        log.record(
            RollbackEntry(
                index=1,
                operation_type="move",
                source="/c",
                target="/d",
                executed_at=datetime.now(),
            )
        )

        removed = log.remove_last()

        assert removed is not None
        assert removed.index == 1
        assert len(log.entries) == 1

