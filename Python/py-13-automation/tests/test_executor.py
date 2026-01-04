"""
测试执行器模块
"""

import pytest
from pathlib import Path

from file_automation.operations import Operation, OpType
from file_automation.executor import (
    Executor,
    ExecutionResult,
    ExecutionSummary,
    preview_operations,
)


class TestExecutionResult:
    """测试执行结果"""

    def test_success_str(self, temp_dir: Path):
        """测试成功结果的字符串表示"""
        op = Operation(OpType.RENAME, temp_dir / "a.txt", temp_dir / "b.txt")
        result = ExecutionResult(operation=op, success=True)

        assert "[SUCCESS]" in str(result)

    def test_failed_str(self, temp_dir: Path):
        """测试失败结果的字符串表示"""
        op = Operation(OpType.DELETE, temp_dir / "a.txt")
        result = ExecutionResult(operation=op, success=False, error="Not found")

        assert "[FAILED]" in str(result)
        assert "Not found" in str(result)

    def test_skipped_str(self, temp_dir: Path):
        """测试跳过结果的字符串表示"""
        op = Operation(OpType.MKDIR, temp_dir / "dir")
        result = ExecutionResult(operation=op, success=True, skipped=True)

        assert "[SKIPPED]" in str(result)


class TestExecutionSummary:
    """测试执行摘要"""

    def test_success_rate(self):
        """测试成功率计算"""
        summary = ExecutionSummary(total=10, success=8, failed=2)
        assert summary.success_rate == 80.0

    def test_success_rate_zero_total(self):
        """测试零总数时的成功率"""
        summary = ExecutionSummary(total=0, success=0, failed=0)
        assert summary.success_rate == 0.0

    def test_to_dict(self):
        """测试转换为字典"""
        summary = ExecutionSummary(
            total=10,
            success=7,
            failed=2,
            skipped=1,
            duration_ms=1234.5,
        )
        data = summary.to_dict()

        assert data["total"] == 10
        assert data["success"] == 7
        assert data["failed"] == 2
        assert data["skipped"] == 1
        assert "70.0%" in data["success_rate"]


class TestExecutor:
    """测试执行器"""

    def test_dry_run(self, sample_files: Path):
        """测试 dry-run 模式"""
        operations = [
            Operation(OpType.RENAME, sample_files / "image.jpg", sample_files / "photo.jpg"),
        ]

        executor = Executor(dry_run=True)
        summary = executor.execute(operations)

        assert summary.success == 1
        assert summary.failed == 0
        # 文件应该没有被重命名
        assert (sample_files / "image.jpg").exists()
        assert not (sample_files / "photo.jpg").exists()

    def test_execute_rename(self, sample_files: Path):
        """测试实际执行重命名"""
        src = sample_files / "image.jpg"
        dst = sample_files / "photo.jpg"

        operations = [Operation(OpType.RENAME, src, dst)]

        executor = Executor(dry_run=False)
        summary = executor.execute(operations)

        assert summary.success == 1
        assert not src.exists()
        assert dst.exists()

    def test_execute_with_state(self, sample_files: Path, state_file: Path):
        """测试带状态管理的执行"""
        operations = [
            Operation(OpType.MKDIR, sample_files / "new_dir1"),
            Operation(OpType.MKDIR, sample_files / "new_dir2"),
        ]

        executor = Executor(dry_run=False, state_file=state_file)
        summary = executor.execute(operations, batch_id="test_batch")

        assert summary.success == 2
        assert state_file.exists()

    def test_skip_completed(self, sample_files: Path, state_file: Path):
        """测试跳过已完成的操作"""
        operations = [
            Operation(OpType.MKDIR, sample_files / "dir1"),
            Operation(OpType.MKDIR, sample_files / "dir2"),
        ]

        # 第一次执行
        executor1 = Executor(dry_run=False, state_file=state_file)
        executor1.execute(operations, batch_id="test")

        # 第二次执行应该跳过
        executor2 = Executor(dry_run=False, state_file=state_file)
        summary = executor2.execute(operations)

        assert summary.skipped == 2
        assert summary.success == 0

    def test_idempotent_skip(self, sample_files: Path):
        """测试幂等跳过"""
        # 目标已存在，源不存在
        src = sample_files / "nonexistent.txt"
        dst = sample_files / "image.jpg"  # 已存在

        operations = [Operation(OpType.RENAME, src, dst)]

        executor = Executor(dry_run=False)
        summary = executor.execute(operations)

        assert summary.skipped == 1

    def test_continue_on_error(self, sample_files: Path):
        """测试错误后继续"""
        operations = [
            Operation(OpType.DELETE, sample_files / "nonexistent.txt"),  # 会失败
            Operation(OpType.MKDIR, sample_files / "new_dir"),  # 应该继续执行
        ]

        executor = Executor(dry_run=False, continue_on_error=True)
        summary = executor.execute(operations)

        assert summary.failed == 1
        assert summary.success == 1
        assert (sample_files / "new_dir").exists()

    def test_stop_on_error(self, sample_files: Path):
        """测试错误后停止"""
        operations = [
            Operation(OpType.DELETE, sample_files / "nonexistent.txt"),  # 会失败
            Operation(OpType.MKDIR, sample_files / "new_dir"),  # 不应该执行
        ]

        executor = Executor(dry_run=False, continue_on_error=False)
        summary = executor.execute(operations)

        assert summary.failed == 1
        assert summary.success == 0
        assert not (sample_files / "new_dir").exists()

    def test_rollback(self, sample_files: Path, rollback_file: Path):
        """测试回滚"""
        src = sample_files / "image.jpg"
        dst = sample_files / "photo.jpg"

        operations = [Operation(OpType.RENAME, src, dst)]

        # 执行
        executor = Executor(dry_run=False, rollback_file=rollback_file)
        executor.execute(operations)

        assert dst.exists()
        assert not src.exists()

        # 回滚
        executor.rollback()

        assert src.exists()
        assert not dst.exists()


class TestPreviewOperations:
    """测试预览操作"""

    def test_preview(self, temp_dir: Path):
        """测试生成预览"""
        operations = [
            Operation(OpType.RENAME, temp_dir / "a.txt", temp_dir / "b.txt"),
            Operation(OpType.DELETE, temp_dir / "c.txt"),
            Operation(OpType.MKDIR, temp_dir / "new_dir"),
        ]

        preview = preview_operations(operations)

        assert "操作预览" in preview
        assert "总计: 3 个操作" in preview
        assert "RENAME" in preview
        assert "DELETE" in preview
        assert "MKDIR" in preview

