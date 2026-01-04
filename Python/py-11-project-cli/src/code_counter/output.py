"""输出格式化

支持表格、JSON、Markdown 格式。
"""

import json
from dataclasses import asdict
from typing import Literal

from code_counter.models import ScanResult

OutputFormat = Literal["table", "json", "markdown"]


class OutputFormatter:
    """输出格式化器"""

    def format(self, result: ScanResult, fmt: OutputFormat = "table") -> str:
        """格式化输出

        Args:
            result: 扫描结果
            fmt: 输出格式

        Returns:
            格式化后的字符串
        """
        if fmt == "json":
            return self.to_json(result)
        elif fmt == "markdown":
            return self.to_markdown(result)
        else:
            return self.to_table(result)

    def to_table(self, result: ScanResult) -> str:
        """转换为表格格式"""
        lines = []

        # 表头
        header = f"{'Language':<15} {'Files':>8} {'Code':>10} {'Comments':>10} {'Blank':>8} {'Total':>10}"
        separator = "-" * len(header)

        lines.append(separator)
        lines.append(header)
        lines.append(separator)

        # 按代码行数排序
        sorted_langs = sorted(
            result.by_language.values(),
            key=lambda x: x.code_lines,
            reverse=True,
        )

        for stats in sorted_langs:
            line = (
                f"{stats.language:<15} "
                f"{stats.file_count:>8,} "
                f"{stats.code_lines:>10,} "
                f"{stats.comment_lines:>10,} "
                f"{stats.blank_lines:>8,} "
                f"{stats.total_lines:>10,}"
            )
            lines.append(line)

        lines.append(separator)

        # 总计
        total_line = (
            f"{'Total':<15} "
            f"{result.total_files:>8,} "
            f"{result.total_code_lines:>10,} "
            f"{result.total_comment_lines:>10,} "
            f"{result.total_blank_lines:>8,} "
            f"{result.total_lines:>10,}"
        )
        lines.append(total_line)
        lines.append(separator)

        # 统计信息
        if result.excluded_count > 0:
            lines.append(f"\nExcluded: {result.excluded_count} files")
        if result.error_count > 0:
            lines.append(f"Errors: {result.error_count}")

        return "\n".join(lines)

    def to_json(self, result: ScanResult) -> str:
        """转换为 JSON 格式"""
        data = {
            "summary": {
                "root": str(result.root_path),
                "total_files": result.total_files,
                "total_lines": result.total_lines,
                "code_lines": result.total_code_lines,
                "comment_lines": result.total_comment_lines,
                "blank_lines": result.total_blank_lines,
            },
            "by_language": {
                lang: {
                    "files": stats.file_count,
                    "code": stats.code_lines,
                    "comments": stats.comment_lines,
                    "blank": stats.blank_lines,
                    "total": stats.total_lines,
                }
                for lang, stats in result.by_language.items()
            },
            "files": [
                {
                    "path": str(f.path),
                    "language": f.language,
                    "code": f.code_lines,
                    "comments": f.comment_lines,
                    "blank": f.blank_lines,
                    "total": f.total_lines,
                }
                for f in result.files
            ],
        }

        if result.errors:
            data["errors"] = result.errors

        return json.dumps(data, indent=2, ensure_ascii=False)

    def to_markdown(self, result: ScanResult) -> str:
        """转换为 Markdown 格式"""
        lines = [
            f"# Code Statistics for `{result.root_path.name}`",
            "",
            "## Summary",
            "",
            f"- **Total Files**: {result.total_files:,}",
            f"- **Total Lines**: {result.total_lines:,}",
            f"- **Code Lines**: {result.total_code_lines:,}",
            f"- **Comment Lines**: {result.total_comment_lines:,}",
            f"- **Blank Lines**: {result.total_blank_lines:,}",
            "",
            "## By Language",
            "",
            "| Language | Files | Code | Comments | Blank | Total |",
            "|----------|------:|-----:|---------:|------:|------:|",
        ]

        # 按代码行数排序
        sorted_langs = sorted(
            result.by_language.values(),
            key=lambda x: x.code_lines,
            reverse=True,
        )

        for stats in sorted_langs:
            line = (
                f"| {stats.language} "
                f"| {stats.file_count:,} "
                f"| {stats.code_lines:,} "
                f"| {stats.comment_lines:,} "
                f"| {stats.blank_lines:,} "
                f"| {stats.total_lines:,} |"
            )
            lines.append(line)

        # 总计行
        lines.append(
            f"| **Total** "
            f"| **{result.total_files:,}** "
            f"| **{result.total_code_lines:,}** "
            f"| **{result.total_comment_lines:,}** "
            f"| **{result.total_blank_lines:,}** "
            f"| **{result.total_lines:,}** |"
        )

        lines.append("")

        # 错误信息
        if result.errors:
            lines.append("## Errors")
            lines.append("")
            for error in result.errors:
                lines.append(f"- {error}")
            lines.append("")

        return "\n".join(lines)


def format_output(result: ScanResult, fmt: OutputFormat = "table") -> str:
    """便捷函数：格式化输出"""
    formatter = OutputFormatter()
    return formatter.format(result, fmt)

