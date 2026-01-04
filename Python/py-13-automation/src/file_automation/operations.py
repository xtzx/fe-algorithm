"""
原子操作定义

定义所有支持的文件操作类型和操作数据结构
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import shutil


class OpType(Enum):
    """操作类型"""

    RENAME = "rename"
    MOVE = "move"
    COPY = "copy"
    DELETE = "delete"
    MKDIR = "mkdir"


@dataclass
class Operation:
    """
    单个原子操作

    Attributes:
        op_type: 操作类型
        source: 源路径
        target: 目标路径（部分操作可选）
        metadata: 附加元数据
    """

    op_type: OpType
    source: Path
    target: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        match self.op_type:
            case OpType.RENAME:
                return f"RENAME: {self.source.name} → {self.target.name if self.target else '?'}"
            case OpType.MOVE:
                return f"MOVE: {self.source} → {self.target}"
            case OpType.COPY:
                return f"COPY: {self.source} → {self.target}"
            case OpType.DELETE:
                return f"DELETE: {self.source}"
            case OpType.MKDIR:
                return f"MKDIR: {self.source}"

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "type": self.op_type.value,
            "source": str(self.source),
            "target": str(self.target) if self.target else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Operation":
        """从字典反序列化"""
        return cls(
            op_type=OpType(data["type"]),
            source=Path(data["source"]),
            target=Path(data["target"]) if data.get("target") else None,
            metadata=data.get("metadata", {}),
        )

    def get_rollback(self) -> "Operation | None":
        """获取回滚操作"""
        match self.op_type:
            case OpType.RENAME:
                if self.target:
                    return Operation(
                        op_type=OpType.RENAME,
                        source=self.target,
                        target=self.source,
                        metadata={"rollback_of": self.to_dict()},
                    )
            case OpType.MOVE:
                if self.target:
                    return Operation(
                        op_type=OpType.MOVE,
                        source=self.target,
                        target=self.source,
                        metadata={"rollback_of": self.to_dict()},
                    )
            case OpType.COPY:
                if self.target:
                    return Operation(
                        op_type=OpType.DELETE,
                        source=self.target,
                        metadata={"rollback_of": self.to_dict()},
                    )
            case OpType.DELETE:
                # 删除操作需要备份才能回滚
                backup_path = self.metadata.get("backup_path")
                if backup_path:
                    return Operation(
                        op_type=OpType.MOVE,
                        source=Path(backup_path),
                        target=self.source,
                        metadata={"rollback_of": self.to_dict()},
                    )
            case OpType.MKDIR:
                return Operation(
                    op_type=OpType.DELETE,
                    source=self.source,
                    metadata={"rollback_of": self.to_dict()},
                )
        return None


class OperationExecutor:
    """操作执行器"""

    @staticmethod
    def execute(op: Operation) -> bool:
        """
        执行单个操作

        Returns:
            True 如果成功，False 如果失败
        """
        try:
            match op.op_type:
                case OpType.RENAME:
                    if op.target is None:
                        raise ValueError("RENAME 操作需要 target")
                    op.source.rename(op.target)

                case OpType.MOVE:
                    if op.target is None:
                        raise ValueError("MOVE 操作需要 target")
                    op.target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(op.source, op.target)

                case OpType.COPY:
                    if op.target is None:
                        raise ValueError("COPY 操作需要 target")
                    op.target.parent.mkdir(parents=True, exist_ok=True)
                    if op.source.is_dir():
                        shutil.copytree(op.source, op.target)
                    else:
                        shutil.copy2(op.source, op.target)

                case OpType.DELETE:
                    if op.source.is_dir():
                        shutil.rmtree(op.source)
                    else:
                        op.source.unlink()

                case OpType.MKDIR:
                    op.source.mkdir(parents=True, exist_ok=True)

            return True

        except Exception:
            return False

    @staticmethod
    def is_idempotent_complete(op: Operation) -> bool:
        """
        检查操作是否已经完成（幂等检查）

        用于断点续跑时判断是否需要执行
        """
        match op.op_type:
            case OpType.RENAME | OpType.MOVE:
                # 目标存在且源不存在 = 已完成
                if op.target and op.target.exists() and not op.source.exists():
                    return True

            case OpType.COPY:
                # 目标存在 = 已完成（简化检查，不验证内容）
                if op.target and op.target.exists():
                    return True

            case OpType.DELETE:
                # 源不存在 = 已删除
                if not op.source.exists():
                    return True

            case OpType.MKDIR:
                # 目录存在 = 已创建
                if op.source.exists() and op.source.is_dir():
                    return True

        return False

