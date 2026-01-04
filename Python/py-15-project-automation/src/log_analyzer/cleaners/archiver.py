"""
日志归档器

支持功能：
- 按时间归档
- dry-run 预览
- 断点续跑
"""

import json
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from log_analyzer.models import CleanupAction, CleanupResult, CleanupState, CleanupTask


class LogArchiver:
    """日志归档器"""

    def __init__(
        self,
        archive_dir: Path,
        state_file: Path | None = None,
    ) -> None:
        self.archive_dir = Path(archive_dir)
        self.state_file = state_file or Path(".log-archiver-state.json")
        self.tasks: list[CleanupTask] = []
        self.state: CleanupState | None = None

    def plan(
        self,
        source_dir: Path,
        older_than_days: int = 30,
        pattern: str = "*.log",
    ) -> list[CleanupTask]:
        """
        规划归档任务

        Args:
            source_dir: 源目录
            older_than_days: 归档超过指定天数的文件
            pattern: 文件匹配模式

        Returns:
            归档任务列表
        """
        self.tasks = []
        cutoff_date = datetime.now() - timedelta(days=older_than_days)

        for file_path in Path(source_dir).glob(pattern):
            if not file_path.is_file():
                continue

            # 检查文件修改时间
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime >= cutoff_date:
                continue

            # 生成归档目标路径
            year_month = mtime.strftime("%Y-%m")
            target_dir = self.archive_dir / year_month
            target_path = target_dir / file_path.name

            task = CleanupTask(
                source_path=str(file_path),
                action=CleanupAction.ARCHIVE,
                target_path=str(target_path),
                size_bytes=file_path.stat().st_size,
                modified_time=mtime,
            )
            self.tasks.append(task)

        return self.tasks

    def preview(self) -> str:
        """
        预览归档任务（dry-run）

        Returns:
            预览文本
        """
        if not self.tasks:
            return "No files to archive"

        lines = ["=" * 60, "Archive Preview (dry-run)", "=" * 60, ""]

        total_size = 0
        for task in self.tasks:
            size_mb = task.size_bytes / (1024 * 1024)
            total_size += task.size_bytes
            lines.append(f"  {task.source_path}")
            lines.append(f"    → {task.target_path} ({size_mb:.2f} MB)")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"Total: {len(self.tasks)} files, {total_size / (1024*1024):.2f} MB")

        return "\n".join(lines)

    def execute(
        self,
        dry_run: bool = False,
        resume: bool = False,
    ) -> CleanupResult:
        """
        执行归档

        Args:
            dry_run: 仅预览，不实际执行
            resume: 是否恢复之前中断的任务

        Returns:
            清理结果
        """
        if dry_run:
            print(self.preview())
            return CleanupResult(total_files=len(self.tasks))

        result = CleanupResult(total_files=len(self.tasks))

        # 恢复之前的状态
        if resume and self.state_file.exists():
            self._load_state()

        # 初始化新状态
        if self.state is None:
            self.state = CleanupState(
                batch_id=str(uuid.uuid4())[:8],
                created_at=datetime.now(),
                total_tasks=len(self.tasks),
                pending=list(range(len(self.tasks))),
            )

        # 执行任务
        for idx in list(self.state.pending):
            if idx >= len(self.tasks):
                continue

            task = self.tasks[idx]
            try:
                self._execute_task(task)
                self.state.pending.remove(idx)
                self.state.completed.append(idx)
                result.processed_files += 1
                result.bytes_archived += task.size_bytes
            except Exception as e:
                self.state.pending.remove(idx)
                self.state.failed.append(idx)
                result.failed_files += 1
                result.errors.append(f"{task.source_path}: {e}")

            # 保存状态（断点续跑）
            self._save_state()

        # 完成后清理状态文件
        if not self.state.pending:
            self._clear_state()

        return result

    def _execute_task(self, task: CleanupTask) -> None:
        """执行单个归档任务"""
        source = Path(task.source_path)
        target = Path(task.target_path) if task.target_path else None

        if target is None:
            raise ValueError("Target path is required for archive action")

        # 确保目标目录存在
        target.parent.mkdir(parents=True, exist_ok=True)

        # 移动文件
        shutil.move(str(source), str(target))

    def _save_state(self) -> None:
        """保存状态"""
        if self.state:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state.model_dump(mode="json"), f, indent=2, default=str)

    def _load_state(self) -> None:
        """加载状态"""
        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)
                self.state = CleanupState.model_validate(data)
        except Exception:
            self.state = None

    def _clear_state(self) -> None:
        """清理状态文件"""
        if self.state_file.exists():
            self.state_file.unlink()
        self.state = None

    def get_status(self) -> str:
        """获取当前状态"""
        if not self.state_file.exists():
            return "No active cleanup job"

        self._load_state()
        if self.state is None:
            return "No active cleanup job"

        lines = [
            f"Batch ID: {self.state.batch_id}",
            f"Created: {self.state.created_at}",
            f"Total: {self.state.total_tasks}",
            f"Completed: {len(self.state.completed)}",
            f"Failed: {len(self.state.failed)}",
            f"Pending: {len(self.state.pending)}",
        ]

        return "\n".join(lines)

