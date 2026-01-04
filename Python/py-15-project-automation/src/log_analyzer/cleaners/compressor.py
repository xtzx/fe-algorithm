"""
日志压缩器

支持 gzip 和 tar.gz 压缩
"""

import gzip
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

from log_analyzer.models import CleanupAction, CleanupResult, CleanupTask


class LogCompressor:
    """日志压缩器"""

    def __init__(
        self,
        compress_level: int = 6,
        delete_original: bool = True,
    ) -> None:
        self.compress_level = compress_level
        self.delete_original = delete_original
        self.tasks: list[CleanupTask] = []

    def plan(
        self,
        source_dir: Path,
        older_than_days: int = 30,
        pattern: str = "*.log",
    ) -> list[CleanupTask]:
        """
        规划压缩任务

        Args:
            source_dir: 源目录
            older_than_days: 压缩超过指定天数的文件
            pattern: 文件匹配模式

        Returns:
            压缩任务列表
        """
        self.tasks = []
        cutoff_date = datetime.now() - timedelta(days=older_than_days)

        for file_path in Path(source_dir).glob(pattern):
            if not file_path.is_file():
                continue

            # 跳过已压缩的文件
            if file_path.suffix in (".gz", ".tar"):
                continue

            # 检查文件修改时间
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime >= cutoff_date:
                continue

            target_path = str(file_path) + ".gz"

            task = CleanupTask(
                source_path=str(file_path),
                action=CleanupAction.COMPRESS,
                target_path=target_path,
                size_bytes=file_path.stat().st_size,
                modified_time=mtime,
            )
            self.tasks.append(task)

        return self.tasks

    def preview(self) -> str:
        """
        预览压缩任务（dry-run）

        Returns:
            预览文本
        """
        if not self.tasks:
            return "No files to compress"

        lines = ["=" * 60, "Compress Preview (dry-run)", "=" * 60, ""]

        total_size = 0
        for task in self.tasks:
            size_mb = task.size_bytes / (1024 * 1024)
            total_size += task.size_bytes
            lines.append(f"  {task.source_path} ({size_mb:.2f} MB)")
            lines.append(f"    → {task.target_path}")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"Total: {len(self.tasks)} files, {total_size / (1024*1024):.2f} MB")
        lines.append("(Estimated compression: ~10x for text logs)")

        return "\n".join(lines)

    def execute(self, dry_run: bool = False) -> CleanupResult:
        """
        执行压缩

        Args:
            dry_run: 仅预览，不实际执行

        Returns:
            清理结果
        """
        if dry_run:
            print(self.preview())
            return CleanupResult(total_files=len(self.tasks))

        result = CleanupResult(total_files=len(self.tasks))

        for task in self.tasks:
            try:
                self._compress_file(task)
                result.processed_files += 1
                result.bytes_freed += task.size_bytes
            except Exception as e:
                result.failed_files += 1
                result.errors.append(f"{task.source_path}: {e}")

        return result

    def _compress_file(self, task: CleanupTask) -> None:
        """压缩单个文件"""
        source = Path(task.source_path)
        target = Path(task.target_path) if task.target_path else None

        if target is None:
            target = Path(str(source) + ".gz")

        # 使用 gzip 压缩
        with open(source, "rb") as f_in:
            with gzip.open(target, "wb", compresslevel=self.compress_level) as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 删除原文件
        if self.delete_original:
            source.unlink()

    def compress_to_tar(
        self,
        source_dir: Path,
        output_path: Path,
        pattern: str = "*.log",
    ) -> Path:
        """
        将多个日志文件压缩到一个 tar.gz

        Args:
            source_dir: 源目录
            output_path: 输出路径
            pattern: 文件匹配模式

        Returns:
            tar.gz 文件路径
        """
        output_path = Path(output_path)

        with tarfile.open(output_path, "w:gz") as tar:
            for file_path in Path(source_dir).glob(pattern):
                if file_path.is_file():
                    tar.add(file_path, arcname=file_path.name)

        return output_path

