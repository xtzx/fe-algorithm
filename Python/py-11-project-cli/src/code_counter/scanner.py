"""文件扫描器

递归扫描目录，支持排除规则。
"""

import fnmatch
import logging
from pathlib import Path

from code_counter.models import get_language

logger = logging.getLogger(__name__)


class Scanner:
    """文件扫描器

    递归扫描目录中的源代码文件。
    """

    # 默认排除的目录
    DEFAULT_EXCLUDES: list[str] = [
        ".git",
        ".svn",
        ".hg",
        ".bzr",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
        ".tox",
        ".nox",
        "htmlcov",
        ".coverage",
    ]

    def __init__(
        self,
        root: Path,
        exclude_patterns: list[str] | None = None,
        include_hidden: bool = False,
        use_gitignore: bool = True,
    ):
        """初始化扫描器

        Args:
            root: 根目录
            exclude_patterns: 额外的排除模式
            include_hidden: 是否包含隐藏文件
            use_gitignore: 是否读取 .gitignore
        """
        self.root = root.resolve()
        self.include_hidden = include_hidden
        self.use_gitignore = use_gitignore

        # 合并排除模式
        self.exclude_patterns = list(self.DEFAULT_EXCLUDES)
        if exclude_patterns:
            self.exclude_patterns.extend(exclude_patterns)

        # 加载 .gitignore
        self._gitignore_patterns: list[str] = []
        if use_gitignore:
            self._load_gitignore()

    def _load_gitignore(self) -> None:
        """加载 .gitignore 文件"""
        gitignore_path = self.root / ".gitignore"
        if not gitignore_path.exists():
            return

        try:
            content = gitignore_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith("#"):
                    continue
                self._gitignore_patterns.append(line)
            logger.debug(f"Loaded {len(self._gitignore_patterns)} patterns from .gitignore")
        except Exception as e:
            logger.warning(f"Failed to read .gitignore: {e}")

    def _should_exclude(self, path: Path) -> bool:
        """检查是否应该排除

        Args:
            path: 文件或目录路径

        Returns:
            True 如果应该排除
        """
        name = path.name

        # 隐藏文件
        if not self.include_hidden and name.startswith("."):
            return True

        # 检查排除模式
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True

        # 检查 .gitignore 模式
        try:
            rel_path = path.relative_to(self.root)
            rel_str = str(rel_path)
        except ValueError:
            rel_str = name

        for pattern in self._gitignore_patterns:
            # 处理目录模式（以 / 结尾）
            if pattern.endswith("/"):
                if path.is_dir() and fnmatch.fnmatch(name, pattern.rstrip("/")):
                    return True
            else:
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_str, pattern):
                    return True

        return False

    def scan(self) -> list[Path]:
        """扫描目录

        Returns:
            找到的源代码文件列表
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")

        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.root}")

        files: list[Path] = []
        excluded_count = 0

        for path in self._walk(self.root):
            if path.is_file():
                # 检查是否是代码文件
                language = get_language(path)
                if language is not None:
                    files.append(path)
                    logger.debug(f"Found: {path} ({language})")
                else:
                    excluded_count += 1
                    logger.debug(f"Skipped (unknown type): {path}")

        logger.info(f"Scanned {len(files)} files, excluded {excluded_count}")
        return files

    def _walk(self, directory: Path):
        """递归遍历目录

        Args:
            directory: 目录路径

        Yields:
            文件和目录路径
        """
        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            logger.warning(f"Permission denied: {directory}")
            return
        except Exception as e:
            logger.warning(f"Error reading directory {directory}: {e}")
            return

        for entry in entries:
            # 检查是否排除
            if self._should_exclude(entry):
                logger.debug(f"Excluded: {entry}")
                continue

            yield entry

            # 递归目录
            if entry.is_dir():
                yield from self._walk(entry)

    def get_excluded_patterns(self) -> list[str]:
        """获取所有排除模式"""
        return self.exclude_patterns + self._gitignore_patterns

