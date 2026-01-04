"""行数统计器

统计代码行、注释行、空行。
"""

import logging
import re
from pathlib import Path

from code_counter.models import FileStats, ScanResult, get_language

logger = logging.getLogger(__name__)


# 语言的注释模式
COMMENT_PATTERNS: dict[str, dict[str, str | tuple[str, str]]] = {
    "Python": {
        "line": "#",
        "block": ('"""', '"""'),
        "block_alt": ("'''", "'''"),
    },
    "JavaScript": {
        "line": "//",
        "block": ("/*", "*/"),
    },
    "TypeScript": {
        "line": "//",
        "block": ("/*", "*/"),
    },
    "Java": {
        "line": "//",
        "block": ("/*", "*/"),
    },
    "C": {
        "line": "//",
        "block": ("/*", "*/"),
    },
    "C++": {
        "line": "//",
        "block": ("/*", "*/"),
    },
    "Go": {
        "line": "//",
        "block": ("/*", "*/"),
    },
    "Rust": {
        "line": "//",
        "block": ("/*", "*/"),
    },
    "C#": {
        "line": "//",
        "block": ("/*", "*/"),
    },
    "Ruby": {
        "line": "#",
        "block": ("=begin", "=end"),
    },
    "PHP": {
        "line": "//",
        "line_alt": "#",
        "block": ("/*", "*/"),
    },
    "Shell": {
        "line": "#",
    },
    "HTML": {
        "block": ("<!--", "-->"),
    },
    "CSS": {
        "block": ("/*", "*/"),
    },
    "SQL": {
        "line": "--",
        "block": ("/*", "*/"),
    },
    "YAML": {
        "line": "#",
    },
    "TOML": {
        "line": "#",
    },
    "INI": {
        "line": ";",
        "line_alt": "#",
    },
}


class Counter:
    """代码行数统计器"""

    def __init__(self):
        """初始化统计器"""
        pass

    def count_file(self, path: Path) -> FileStats:
        """统计单个文件

        Args:
            path: 文件路径

        Returns:
            FileStats 对象
        """
        language = get_language(path) or "Unknown"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return FileStats(path=path, language=language)

        lines = content.splitlines()
        total_lines = len(lines)
        blank_lines = 0
        comment_lines = 0
        code_lines = 0

        # 获取语言的注释模式
        patterns = COMMENT_PATTERNS.get(language, {})
        line_comment = patterns.get("line", "")
        line_comment_alt = patterns.get("line_alt", "")
        block_start, block_end = patterns.get("block", ("", ""))
        block_start_alt, block_end_alt = patterns.get("block_alt", ("", ""))

        in_block_comment = False
        in_block_alt = False

        for line in lines:
            stripped = line.strip()

            # 空行
            if not stripped:
                blank_lines += 1
                continue

            # 块注释处理
            if in_block_comment:
                comment_lines += 1
                if block_end and block_end in stripped:
                    in_block_comment = False
                continue

            if in_block_alt:
                comment_lines += 1
                if block_end_alt and block_end_alt in stripped:
                    in_block_alt = False
                continue

            # 检查块注释开始
            if block_start and stripped.startswith(block_start):
                comment_lines += 1
                if block_end not in stripped[len(block_start) :]:
                    in_block_comment = True
                continue

            if block_start_alt and stripped.startswith(block_start_alt):
                comment_lines += 1
                if block_end_alt not in stripped[len(block_start_alt) :]:
                    in_block_alt = True
                continue

            # 行注释
            if line_comment and stripped.startswith(line_comment):
                comment_lines += 1
                continue

            if line_comment_alt and stripped.startswith(line_comment_alt):
                comment_lines += 1
                continue

            # 代码行
            code_lines += 1

        return FileStats(
            path=path,
            language=language,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
        )

    def count_files(self, files: list[Path], root: Path) -> ScanResult:
        """统计多个文件

        Args:
            files: 文件列表
            root: 根目录

        Returns:
            ScanResult 对象
        """
        result = ScanResult(root_path=root)

        for path in files:
            try:
                stats = self.count_file(path)
                result.add_file(stats)
                logger.debug(
                    f"Counted {path}: {stats.code_lines} code, "
                    f"{stats.comment_lines} comments, {stats.blank_lines} blank"
                )
            except Exception as e:
                error_msg = f"Error counting {path}: {e}"
                result.add_error(error_msg)
                logger.warning(error_msg)

        return result


def count_directory(
    path: Path,
    exclude_patterns: list[str] | None = None,
    use_gitignore: bool = True,
) -> ScanResult:
    """统计目录中的代码

    便捷函数，结合 Scanner 和 Counter。

    Args:
        path: 目录路径
        exclude_patterns: 排除模式
        use_gitignore: 是否使用 .gitignore

    Returns:
        ScanResult 对象
    """
    from code_counter.scanner import Scanner

    scanner = Scanner(
        root=path,
        exclude_patterns=exclude_patterns,
        use_gitignore=use_gitignore,
    )
    files = scanner.scan()

    counter = Counter()
    return counter.count_files(files, path)

