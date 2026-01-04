"""数据模型

使用 dataclass 定义所有数据结构。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileStats:
    """单个文件的统计信息"""

    path: Path
    language: str
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0

    def __post_init__(self) -> None:
        """验证数据"""
        if self.total_lines < 0:
            raise ValueError("total_lines cannot be negative")
        if self.code_lines < 0:
            raise ValueError("code_lines cannot be negative")
        if self.comment_lines < 0:
            raise ValueError("comment_lines cannot be negative")
        if self.blank_lines < 0:
            raise ValueError("blank_lines cannot be negative")


@dataclass
class LanguageStats:
    """按语言汇总的统计信息"""

    language: str
    file_count: int = 0
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0

    def add_file(self, file_stats: FileStats) -> None:
        """添加一个文件的统计"""
        self.file_count += 1
        self.total_lines += file_stats.total_lines
        self.code_lines += file_stats.code_lines
        self.comment_lines += file_stats.comment_lines
        self.blank_lines += file_stats.blank_lines


@dataclass
class ScanResult:
    """扫描结果"""

    root_path: Path
    files: list[FileStats] = field(default_factory=list)
    by_language: dict[str, LanguageStats] = field(default_factory=dict)
    excluded_count: int = 0
    error_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        """总文件数"""
        return len(self.files)

    @property
    def total_lines(self) -> int:
        """总行数"""
        return sum(f.total_lines for f in self.files)

    @property
    def total_code_lines(self) -> int:
        """总代码行数"""
        return sum(f.code_lines for f in self.files)

    @property
    def total_comment_lines(self) -> int:
        """总注释行数"""
        return sum(f.comment_lines for f in self.files)

    @property
    def total_blank_lines(self) -> int:
        """总空行数"""
        return sum(f.blank_lines for f in self.files)

    def add_file(self, file_stats: FileStats) -> None:
        """添加文件统计"""
        self.files.append(file_stats)

        # 更新语言统计
        lang = file_stats.language
        if lang not in self.by_language:
            self.by_language[lang] = LanguageStats(language=lang)
        self.by_language[lang].add_file(file_stats)

    def add_error(self, error: str) -> None:
        """添加错误"""
        self.errors.append(error)
        self.error_count += 1


# 语言扩展名映射
LANGUAGE_EXTENSIONS: dict[str, str] = {
    # Python
    ".py": "Python",
    ".pyi": "Python",
    ".pyx": "Python",
    # JavaScript/TypeScript
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    # Web
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",
    ".less": "Less",
    # Data
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
    # Shell
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".fish": "Shell",
    # Systems
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".go": "Go",
    ".rs": "Rust",
    # JVM
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".scala": "Scala",
    # .NET
    ".cs": "C#",
    ".fs": "F#",
    # Ruby/PHP
    ".rb": "Ruby",
    ".php": "PHP",
    # Documentation
    ".md": "Markdown",
    ".rst": "reStructuredText",
    ".txt": "Text",
    # Config
    ".ini": "INI",
    ".cfg": "INI",
    ".conf": "Config",
    # SQL
    ".sql": "SQL",
    # Other
    ".r": "R",
    ".R": "R",
    ".swift": "Swift",
    ".dart": "Dart",
    ".lua": "Lua",
    ".vim": "Vim",
    ".dockerfile": "Dockerfile",
    ".makefile": "Makefile",
}


def get_language(path: Path) -> str | None:
    """根据文件扩展名获取语言

    Args:
        path: 文件路径

    Returns:
        语言名称，如果无法识别返回 None
    """
    # 特殊文件名
    name = path.name.lower()
    if name == "dockerfile":
        return "Dockerfile"
    if name == "makefile":
        return "Makefile"
    if name == ".gitignore":
        return "Git"
    if name == ".dockerignore":
        return "Docker"

    # 按扩展名
    suffix = path.suffix.lower()
    return LANGUAGE_EXTENSIONS.get(suffix)

