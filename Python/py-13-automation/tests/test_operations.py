"""
测试操作模块
"""

import pytest
from pathlib import Path

from file_automation.operations import Operation, OpType, OperationExecutor


class TestOperation:
    """测试 Operation 类"""

    def test_rename_str(self, temp_dir: Path):
        """测试重命名操作的字符串表示"""
        op = Operation(
            op_type=OpType.RENAME,
            source=temp_dir / "old.txt",
            target=temp_dir / "new.txt",
        )
        assert "RENAME" in str(op)
        assert "old.txt" in str(op)
        assert "new.txt" in str(op)

    def test_delete_str(self, temp_dir: Path):
        """测试删除操作的字符串表示"""
        op = Operation(
            op_type=OpType.DELETE,
            source=temp_dir / "file.txt",
        )
        assert "DELETE" in str(op)
        assert "file.txt" in str(op)

    def test_to_dict(self, temp_dir: Path):
        """测试序列化"""
        op = Operation(
            op_type=OpType.MOVE,
            source=temp_dir / "src.txt",
            target=temp_dir / "dst.txt",
            metadata={"key": "value"},
        )
        data = op.to_dict()

        assert data["type"] == "move"
        assert "src.txt" in data["source"]
        assert "dst.txt" in data["target"]
        assert data["metadata"] == {"key": "value"}

    def test_from_dict(self, temp_dir: Path):
        """测试反序列化"""
        data = {
            "type": "copy",
            "source": str(temp_dir / "src.txt"),
            "target": str(temp_dir / "dst.txt"),
            "metadata": {},
        }
        op = Operation.from_dict(data)

        assert op.op_type == OpType.COPY
        assert op.source == temp_dir / "src.txt"
        assert op.target == temp_dir / "dst.txt"

    def test_get_rollback_rename(self, temp_dir: Path):
        """测试重命名的回滚操作"""
        op = Operation(
            op_type=OpType.RENAME,
            source=temp_dir / "old.txt",
            target=temp_dir / "new.txt",
        )
        rollback = op.get_rollback()

        assert rollback is not None
        assert rollback.op_type == OpType.RENAME
        assert rollback.source == temp_dir / "new.txt"
        assert rollback.target == temp_dir / "old.txt"

    def test_get_rollback_copy(self, temp_dir: Path):
        """测试复制的回滚操作（删除目标）"""
        op = Operation(
            op_type=OpType.COPY,
            source=temp_dir / "src.txt",
            target=temp_dir / "dst.txt",
        )
        rollback = op.get_rollback()

        assert rollback is not None
        assert rollback.op_type == OpType.DELETE
        assert rollback.source == temp_dir / "dst.txt"


class TestOperationExecutor:
    """测试操作执行器"""

    def test_execute_rename(self, temp_dir: Path):
        """测试执行重命名"""
        src = temp_dir / "old.txt"
        dst = temp_dir / "new.txt"
        src.write_text("content")

        op = Operation(op_type=OpType.RENAME, source=src, target=dst)
        result = OperationExecutor.execute(op)

        assert result is True
        assert not src.exists()
        assert dst.exists()
        assert dst.read_text() == "content"

    def test_execute_copy(self, temp_dir: Path):
        """测试执行复制"""
        src = temp_dir / "src.txt"
        dst = temp_dir / "dst.txt"
        src.write_text("content")

        op = Operation(op_type=OpType.COPY, source=src, target=dst)
        result = OperationExecutor.execute(op)

        assert result is True
        assert src.exists()
        assert dst.exists()
        assert dst.read_text() == "content"

    def test_execute_delete(self, temp_dir: Path):
        """测试执行删除"""
        file_path = temp_dir / "to_delete.txt"
        file_path.write_text("content")

        op = Operation(op_type=OpType.DELETE, source=file_path)
        result = OperationExecutor.execute(op)

        assert result is True
        assert not file_path.exists()

    def test_execute_mkdir(self, temp_dir: Path):
        """测试执行创建目录"""
        dir_path = temp_dir / "new_dir" / "sub_dir"

        op = Operation(op_type=OpType.MKDIR, source=dir_path)
        result = OperationExecutor.execute(op)

        assert result is True
        assert dir_path.exists()
        assert dir_path.is_dir()

    def test_idempotent_complete_rename(self, temp_dir: Path):
        """测试幂等检查 - 重命名已完成"""
        src = temp_dir / "old.txt"
        dst = temp_dir / "new.txt"
        dst.write_text("content")  # 目标存在，源不存在

        op = Operation(op_type=OpType.RENAME, source=src, target=dst)
        assert OperationExecutor.is_idempotent_complete(op) is True

    def test_idempotent_complete_delete(self, temp_dir: Path):
        """测试幂等检查 - 删除已完成"""
        file_path = temp_dir / "deleted.txt"
        # 文件不存在

        op = Operation(op_type=OpType.DELETE, source=file_path)
        assert OperationExecutor.is_idempotent_complete(op) is True

    def test_idempotent_not_complete(self, temp_dir: Path):
        """测试幂等检查 - 操作未完成"""
        src = temp_dir / "old.txt"
        dst = temp_dir / "new.txt"
        src.write_text("content")  # 源存在，目标不存在

        op = Operation(op_type=OpType.RENAME, source=src, target=dst)
        assert OperationExecutor.is_idempotent_complete(op) is False

