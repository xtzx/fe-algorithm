"""
操作执行器

负责执行操作计划，支持 dry-run、断点续跑、重试
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
import logging
import time

from file_automation.operations import Operation, OpType, OperationExecutor
from file_automation.state import (
    StateManager,
    RollbackLog,
    RollbackEntry,
    TaskStatus,
)


@dataclass
class ExecutionResult:
    """执行结果"""

    operation: Operation
    success: bool
    error: str | None = None
    skipped: bool = False
    duration_ms: float = 0.0

    def __str__(self) -> str:
        if self.skipped:
            return f"[SKIPPED] {self.operation}"
        elif self.success:
            return f"[SUCCESS] {self.operation}"
        else:
            return f"[FAILED] {self.operation}: {self.error}"


@dataclass
class ExecutionSummary:
    """执行摘要"""

    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: float = 0.0
    results: list[ExecutionResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success / self.total * 100

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": f"{self.success_rate:.1f}%",
            "duration_ms": self.duration_ms,
        }


class Executor:
    """操作执行器"""

    def __init__(
        self,
        dry_run: bool = False,
        state_file: Path | None = None,
        rollback_file: Path | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        continue_on_error: bool = True,
        logger: logging.Logger | None = None,
    ):
        """
        初始化执行器

        Args:
            dry_run: 是否为预览模式
            state_file: 状态文件路径（用于断点续跑）
            rollback_file: 回滚日志路径
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            continue_on_error: 失败后是否继续
            logger: 日志记录器
        """
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.continue_on_error = continue_on_error
        self.logger = logger or logging.getLogger(__name__)

        # 状态管理
        self.state_manager = StateManager(state_file) if state_file else None
        self.rollback_log = RollbackLog(rollback_file) if rollback_file else None

        # 进度回调
        self.on_progress: Callable[[int, int], None] | None = None

    def execute(
        self,
        operations: list[Operation],
        batch_id: str | None = None,
    ) -> ExecutionSummary:
        """
        执行操作列表

        Args:
            operations: 操作列表
            batch_id: 批次 ID

        Returns:
            执行摘要
        """
        start_time = time.perf_counter()
        summary = ExecutionSummary(total=len(operations))

        # 初始化或加载状态
        if self.state_manager:
            state = self.state_manager.load()
            if state is None:
                batch_id = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
                state = self.state_manager.init_state(batch_id, len(operations))
                self.logger.info(f"开始新批次: {batch_id}")
            else:
                self.logger.info(f"恢复批次: {state.batch_id}")
                self.logger.info(f"  已完成: {state.completed_count}")
                self.logger.info(f"  待执行: {state.pending_count}")

        # 加载回滚日志
        if self.rollback_log:
            self.rollback_log.load()

        # 执行操作
        for i, operation in enumerate(operations):
            # 检查是否需要跳过（已完成）
            if self.state_manager:
                task_state = self.state_manager.state.tasks.get(i)  # type: ignore
                if task_state and task_state.status == TaskStatus.COMPLETED:
                    result = ExecutionResult(
                        operation=operation,
                        success=True,
                        skipped=True,
                    )
                    summary.results.append(result)
                    summary.skipped += 1
                    continue

            # 幂等检查
            if OperationExecutor.is_idempotent_complete(operation):
                result = ExecutionResult(
                    operation=operation,
                    success=True,
                    skipped=True,
                )
                summary.results.append(result)
                summary.skipped += 1
                if self.state_manager:
                    self.state_manager.mark_skipped(i, "Already completed")
                continue

            # 执行操作
            result = self._execute_one(operation, i)
            summary.results.append(result)

            if result.success:
                summary.success += 1
            else:
                summary.failed += 1
                if not self.continue_on_error:
                    self.logger.error("遇到错误，停止执行")
                    break

            # 进度回调
            if self.on_progress:
                self.on_progress(i + 1, len(operations))

        summary.duration_ms = (time.perf_counter() - start_time) * 1000
        return summary

    def _execute_one(self, operation: Operation, index: int) -> ExecutionResult:
        """执行单个操作（带重试）"""
        start_time = time.perf_counter()

        # 标记开始
        if self.state_manager:
            self.state_manager.mark_started(index)

        # Dry-run 模式
        if self.dry_run:
            self.logger.info(f"[DRY-RUN] {operation}")
            if self.state_manager:
                self.state_manager.mark_completed(index)
            return ExecutionResult(
                operation=operation,
                success=True,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # 实际执行（带重试）
        last_error: str | None = None

        for attempt in range(self.max_retries):
            try:
                if OperationExecutor.execute(operation):
                    # 成功
                    duration = (time.perf_counter() - start_time) * 1000
                    self.logger.info(f"[SUCCESS] {operation}")

                    # 记录回滚信息
                    if self.rollback_log:
                        self.rollback_log.record(
                            RollbackEntry(
                                index=index,
                                operation_type=operation.op_type.value,
                                source=str(operation.source),
                                target=str(operation.target) if operation.target else None,
                                executed_at=datetime.now(),
                            )
                        )

                    # 更新状态
                    if self.state_manager:
                        self.state_manager.mark_completed(index)

                    return ExecutionResult(
                        operation=operation,
                        success=True,
                        duration_ms=duration,
                    )
                else:
                    last_error = "Operation returned False"

            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"尝试 {attempt + 1}/{self.max_retries} 失败: {e}"
                )

            # 重试延迟
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        # 所有重试都失败
        duration = (time.perf_counter() - start_time) * 1000
        self.logger.error(f"[FAILED] {operation}: {last_error}")

        if self.state_manager:
            self.state_manager.mark_failed(index, last_error or "Unknown error")

        return ExecutionResult(
            operation=operation,
            success=False,
            error=last_error,
            duration_ms=duration,
        )

    def rollback(self, count: int | None = None) -> int:
        """
        回滚已执行的操作

        Args:
            count: 回滚数量（默认全部）

        Returns:
            成功回滚的操作数量
        """
        if not self.rollback_log:
            self.logger.warning("未配置回滚日志")
            return 0

        self.rollback_log.load()
        entries = self.rollback_log.get_rollback_entries()

        if count is not None:
            entries = entries[:count]

        rolled_back = 0

        for entry in entries:
            # 构造回滚操作
            rollback_op = self._create_rollback_operation(entry)
            if rollback_op is None:
                self.logger.warning(f"无法创建回滚操作: {entry}")
                continue

            if self.dry_run:
                self.logger.info(f"[DRY-RUN ROLLBACK] {rollback_op}")
                rolled_back += 1
            else:
                try:
                    if OperationExecutor.execute(rollback_op):
                        self.logger.info(f"[ROLLBACK] {rollback_op}")
                        self.rollback_log.remove_last()
                        rolled_back += 1
                    else:
                        self.logger.error(f"回滚失败: {rollback_op}")
                except Exception as e:
                    self.logger.error(f"回滚异常: {e}")

        return rolled_back

    def _create_rollback_operation(self, entry: RollbackEntry) -> Operation | None:
        """根据回滚日志条目创建回滚操作"""
        op_type = OpType(entry.operation_type)

        match op_type:
            case OpType.RENAME | OpType.MOVE:
                if entry.target:
                    return Operation(
                        op_type=op_type,
                        source=Path(entry.target),
                        target=Path(entry.source),
                    )
            case OpType.COPY:
                if entry.target:
                    return Operation(
                        op_type=OpType.DELETE,
                        source=Path(entry.target),
                    )
            case OpType.MKDIR:
                return Operation(
                    op_type=OpType.DELETE,
                    source=Path(entry.source),
                )
            # DELETE 操作需要备份才能回滚，这里无法处理

        return None


def preview_operations(
    operations: list[Operation],
    show_details: bool = True,
) -> str:
    """
    预览操作

    Args:
        operations: 操作列表
        show_details: 是否显示详细信息
    """
    lines = [
        "=" * 60,
        "操作预览",
        "=" * 60,
        "",
    ]

    # 按类型分组
    by_type: dict[str, list[Operation]] = {}
    for op in operations:
        by_type.setdefault(op.op_type.value, []).append(op)

    # 统计
    lines.append(f"总计: {len(operations)} 个操作")
    for op_type, ops in by_type.items():
        lines.append(f"  - {op_type.upper()}: {len(ops)}")
    lines.append("")

    # 详细列表
    if show_details:
        lines.append("-" * 60)
        for i, op in enumerate(operations, 1):
            lines.append(f"{i:3d}. {op}")

    lines.append("=" * 60)
    return "\n".join(lines)


def confirm_execution(message: str = "确认执行？", default: bool = False) -> bool:
    """
    确认执行

    Args:
        message: 确认消息
        default: 默认值
    """
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{message} {suffix}: ").strip().lower()

    if not response:
        return default
    return response in ("y", "yes")

