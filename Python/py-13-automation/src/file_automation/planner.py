"""
操作计划器

负责分析需求并生成操作计划
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

from file_automation.operations import Operation, OpType


class Planner(ABC):
    """计划器抽象基类"""

    @abstractmethod
    def create_plan(self) -> list[Operation]:
        """创建操作计划"""
        pass


@dataclass
class RenamePlanner(Planner):
    """重命名计划器"""

    directory: Path

    def create_plan(self) -> list[Operation]:
        """默认实现，返回空计划"""
        return []

    def plan_regex_rename(
        self,
        pattern: str,
        replacement: str,
        file_glob: str = "*",
    ) -> list[Operation]:
        """
        计划正则重命名

        Args:
            pattern: 正则表达式
            replacement: 替换字符串（支持 \\1, \\2 等分组引用）
            file_glob: 文件过滤模式
        """
        operations: list[Operation] = []
        regex = re.compile(pattern)

        for file_path in sorted(self.directory.glob(file_glob)):
            if not file_path.is_file():
                continue

            old_name = file_path.name
            new_name = regex.sub(replacement, old_name)

            if old_name != new_name:
                operations.append(
                    Operation(
                        op_type=OpType.RENAME,
                        source=file_path,
                        target=file_path.parent / new_name,
                        metadata={"old_name": old_name, "new_name": new_name},
                    )
                )

        return operations

    def plan_sequential_rename(
        self,
        prefix: str,
        file_glob: str = "*",
        start: int = 1,
        width: int = 3,
        sort_by: str = "name",  # name, date, size
    ) -> list[Operation]:
        """
        计划序号重命名

        Args:
            prefix: 文件名前缀
            file_glob: 文件过滤模式
            start: 起始序号
            width: 序号宽度（补零）
            sort_by: 排序方式
        """
        operations: list[Operation] = []
        files = [f for f in self.directory.glob(file_glob) if f.is_file()]

        # 排序
        if sort_by == "date":
            files.sort(key=lambda f: f.stat().st_mtime)
        elif sort_by == "size":
            files.sort(key=lambda f: f.stat().st_size)
        else:
            files.sort(key=lambda f: f.name)

        for i, file_path in enumerate(files, start=start):
            suffix = file_path.suffix
            new_name = f"{prefix}{i:0{width}d}{suffix}"

            if file_path.name != new_name:
                operations.append(
                    Operation(
                        op_type=OpType.RENAME,
                        source=file_path,
                        target=file_path.parent / new_name,
                    )
                )

        return operations

    def plan_datetime_rename(
        self,
        file_glob: str = "*",
        date_format: str = "%Y%m%d_%H%M%S",
    ) -> list[Operation]:
        """
        计划日期时间重命名

        Args:
            file_glob: 文件过滤模式
            date_format: 日期格式
        """
        operations: list[Operation] = []
        used_names: set[str] = set()

        for file_path in sorted(self.directory.glob(file_glob)):
            if not file_path.is_file():
                continue

            mtime = file_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            suffix = file_path.suffix

            # 生成新名称
            base_name = dt.strftime(date_format)
            new_name = f"{base_name}{suffix}"

            # 处理重名
            counter = 1
            while new_name in used_names:
                new_name = f"{base_name}_{counter}{suffix}"
                counter += 1

            used_names.add(new_name)

            if file_path.name != new_name:
                operations.append(
                    Operation(
                        op_type=OpType.RENAME,
                        source=file_path,
                        target=file_path.parent / new_name,
                    )
                )

        return operations


# 文件类型映射
FILE_CATEGORIES = {
    "images": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico"],
    "documents": [
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".txt",
        ".md",
        ".rtf",
    ],
    "videos": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"],
    "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"],
    "archives": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"],
    "code": [
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
    ],
    "data": [".json", ".xml", ".csv", ".yaml", ".yml", ".toml", ".sql"],
}


def get_category(suffix: str) -> str:
    """获取文件类别"""
    suffix_lower = suffix.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if suffix_lower in extensions:
            return category
    return "others"


@dataclass
class OrganizePlanner(Planner):
    """文件整理计划器"""

    source_dir: Path
    target_dir: Path | None = None

    def create_plan(self) -> list[Operation]:
        """默认实现，返回空计划"""
        return []

    def plan_by_extension(self) -> list[Operation]:
        """
        计划按扩展名分类
        """
        operations: list[Operation] = []
        target = self.target_dir or self.source_dir

        for file_path in self.source_dir.iterdir():
            if not file_path.is_file():
                continue

            category = get_category(file_path.suffix)
            category_dir = target / category
            new_path = category_dir / file_path.name

            if file_path != new_path:
                # 先创建目录
                if not any(
                    op.source == category_dir and op.op_type == OpType.MKDIR
                    for op in operations
                ):
                    operations.append(
                        Operation(
                            op_type=OpType.MKDIR,
                            source=category_dir,
                        )
                    )

                operations.append(
                    Operation(
                        op_type=OpType.MOVE,
                        source=file_path,
                        target=new_path,
                        metadata={"category": category},
                    )
                )

        return operations

    def plan_by_date(
        self,
        date_format: str = "%Y/%Y-%m",
    ) -> list[Operation]:
        """
        计划按日期分类

        Args:
            date_format: 日期目录格式
        """
        operations: list[Operation] = []
        target = self.target_dir or self.source_dir
        created_dirs: set[Path] = set()

        for file_path in self.source_dir.iterdir():
            if not file_path.is_file():
                continue

            mtime = file_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            date_str = dt.strftime(date_format)

            date_dir = target / date_str
            new_path = date_dir / file_path.name

            if file_path != new_path:
                # 创建目录
                if date_dir not in created_dirs:
                    operations.append(
                        Operation(
                            op_type=OpType.MKDIR,
                            source=date_dir,
                        )
                    )
                    created_dirs.add(date_dir)

                operations.append(
                    Operation(
                        op_type=OpType.MOVE,
                        source=file_path,
                        target=new_path,
                    )
                )

        return operations

    def plan_by_size(self) -> list[Operation]:
        """
        计划按大小分类
        """
        operations: list[Operation] = []
        target = self.target_dir or self.source_dir
        created_dirs: set[Path] = set()

        def get_size_category(size: int) -> str:
            if size < 100 * 1024:  # < 100KB
                return "tiny"
            elif size < 1024 * 1024:  # < 1MB
                return "small"
            elif size < 10 * 1024 * 1024:  # < 10MB
                return "medium"
            elif size < 100 * 1024 * 1024:  # < 100MB
                return "large"
            else:
                return "huge"

        for file_path in self.source_dir.iterdir():
            if not file_path.is_file():
                continue

            size = file_path.stat().st_size
            category = get_size_category(size)

            category_dir = target / category
            new_path = category_dir / file_path.name

            if file_path != new_path:
                if category_dir not in created_dirs:
                    operations.append(
                        Operation(
                            op_type=OpType.MKDIR,
                            source=category_dir,
                        )
                    )
                    created_dirs.add(category_dir)

                operations.append(
                    Operation(
                        op_type=OpType.MOVE,
                        source=file_path,
                        target=new_path,
                        metadata={"size": size, "category": category},
                    )
                )

        return operations


@dataclass
class CleanupPlanner(Planner):
    """清理计划器"""

    directory: Path

    def create_plan(self) -> list[Operation]:
        """默认实现，返回空计划"""
        return []

    def plan_delete_empty_dirs(self) -> list[Operation]:
        """计划删除空目录"""
        operations: list[Operation] = []

        # 从最深层开始
        dirs = sorted(
            [d for d in self.directory.rglob("*") if d.is_dir()],
            key=lambda p: -len(p.parts),
        )

        for dir_path in dirs:
            if not any(dir_path.iterdir()):
                operations.append(
                    Operation(
                        op_type=OpType.DELETE,
                        source=dir_path,
                    )
                )

        return operations

    def plan_delete_by_pattern(
        self,
        patterns: list[str],
    ) -> list[Operation]:
        """
        计划删除匹配模式的文件

        Args:
            patterns: glob 模式列表
        """
        operations: list[Operation] = []
        to_delete: set[Path] = set()

        for pattern in patterns:
            for path in self.directory.rglob(pattern):
                to_delete.add(path)

        for path in to_delete:
            operations.append(
                Operation(
                    op_type=OpType.DELETE,
                    source=path,
                )
            )

        return operations

    def plan_delete_old_files(
        self,
        max_age_days: int,
        file_glob: str = "*",
    ) -> list[Operation]:
        """
        计划删除旧文件

        Args:
            max_age_days: 最大文件年龄（天）
            file_glob: 文件过滤模式
        """
        from datetime import timedelta

        operations: list[Operation] = []
        cutoff = datetime.now() - timedelta(days=max_age_days)

        for file_path in self.directory.rglob(file_glob):
            if not file_path.is_file():
                continue

            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff:
                operations.append(
                    Operation(
                        op_type=OpType.DELETE,
                        source=file_path,
                        metadata={"mtime": mtime.isoformat()},
                    )
                )

        return operations


def analyze_plan(operations: list[Operation]) -> dict:
    """
    分析操作计划

    Returns:
        统计信息
    """
    stats: dict[str, list[Operation]] = defaultdict(list)

    for op in operations:
        stats[op.op_type.value].append(op)

    return {
        "total": len(operations),
        "by_type": {k: len(v) for k, v in stats.items()},
        "operations": stats,
    }

