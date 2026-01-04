"""清理器测试"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from log_analyzer.cleaners import LogArchiver, LogCompressor


class TestLogArchiver:
    """日志归档器测试"""

    def test_plan_archive(self, tmp_path: Path) -> None:
        # 创建测试文件
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        # 创建旧日志（40 天前）
        old_log = log_dir / "old.log"
        old_log.write_text("old log content")
        old_mtime = (datetime.now() - timedelta(days=40)).timestamp()
        import os
        os.utime(old_log, (old_mtime, old_mtime))

        # 创建新日志
        new_log = log_dir / "new.log"
        new_log.write_text("new log content")

        # 规划归档
        archive_dir = tmp_path / "archive"
        archiver = LogArchiver(archive_dir)
        tasks = archiver.plan(log_dir, older_than_days=30)

        # 只有旧日志应该被归档
        assert len(tasks) == 1
        assert "old.log" in tasks[0].source_path

    def test_preview_archive(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        old_log = log_dir / "old.log"
        old_log.write_text("content")
        old_mtime = (datetime.now() - timedelta(days=40)).timestamp()
        import os
        os.utime(old_log, (old_mtime, old_mtime))

        archive_dir = tmp_path / "archive"
        archiver = LogArchiver(archive_dir)
        archiver.plan(log_dir, older_than_days=30)

        preview = archiver.preview()
        assert "old.log" in preview
        assert "Archive Preview" in preview

    def test_execute_archive_dry_run(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        old_log = log_dir / "old.log"
        old_log.write_text("content")

        archive_dir = tmp_path / "archive"
        archiver = LogArchiver(archive_dir)
        archiver.plan(log_dir)

        result = archiver.execute(dry_run=True)
        # 文件不应该被移动
        assert old_log.exists()

    def test_execute_archive(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        old_log = log_dir / "old.log"
        old_log.write_text("content")
        old_mtime = (datetime.now() - timedelta(days=40)).timestamp()
        import os
        os.utime(old_log, (old_mtime, old_mtime))

        archive_dir = tmp_path / "archive"
        state_file = tmp_path / "state.json"
        archiver = LogArchiver(archive_dir, state_file)
        archiver.plan(log_dir, older_than_days=30)

        result = archiver.execute()

        assert result.processed_files == 1
        assert not old_log.exists()
        assert archive_dir.exists()


class TestLogCompressor:
    """日志压缩器测试"""

    def test_plan_compress(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        # 创建旧日志
        old_log = log_dir / "old.log"
        old_log.write_text("old log content")
        old_mtime = (datetime.now() - timedelta(days=40)).timestamp()
        import os
        os.utime(old_log, (old_mtime, old_mtime))

        # 创建新日志
        new_log = log_dir / "new.log"
        new_log.write_text("new log content")

        compressor = LogCompressor()
        tasks = compressor.plan(log_dir, older_than_days=30)

        assert len(tasks) == 1
        assert "old.log" in tasks[0].source_path

    def test_preview_compress(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        old_log = log_dir / "old.log"
        old_log.write_text("content")
        old_mtime = (datetime.now() - timedelta(days=40)).timestamp()
        import os
        os.utime(old_log, (old_mtime, old_mtime))

        compressor = LogCompressor()
        compressor.plan(log_dir, older_than_days=30)

        preview = compressor.preview()
        assert "old.log" in preview
        assert "Compress Preview" in preview

    def test_execute_compress(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        old_log = log_dir / "old.log"
        old_log.write_text("content" * 1000)
        old_mtime = (datetime.now() - timedelta(days=40)).timestamp()
        import os
        os.utime(old_log, (old_mtime, old_mtime))

        compressor = LogCompressor()
        compressor.plan(log_dir, older_than_days=30)

        result = compressor.execute()

        assert result.processed_files == 1
        assert not old_log.exists()
        assert (log_dir / "old.log.gz").exists()

    def test_skip_already_compressed(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        # 创建已压缩的文件
        gz_file = log_dir / "old.log.gz"
        gz_file.write_bytes(b"compressed content")
        old_mtime = (datetime.now() - timedelta(days=40)).timestamp()
        import os
        os.utime(gz_file, (old_mtime, old_mtime))

        compressor = LogCompressor()
        tasks = compressor.plan(log_dir, older_than_days=30)

        # 已压缩的文件应该被跳过
        assert len(tasks) == 0

