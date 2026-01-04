"""
命令行接口

提供 analyze、report、clean 三个子命令
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from log_analyzer.analyzers import ErrorAnalyzer, RequestAnalyzer, TimelineAnalyzer
from log_analyzer.cleaners import LogArchiver, LogCompressor
from log_analyzer.config import get_settings
from log_analyzer.models import AnalysisReport, NginxLogEntry
from log_analyzer.parsers import detect_format, get_parser
from log_analyzer.reporters import JsonReporter, MarkdownReporter, TerminalReporter


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="log-analyzer",
        description="日志分析与清理工具",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # =========================================================================
    # analyze 命令
    # =========================================================================
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="分析日志文件",
    )
    analyze_parser.add_argument(
        "path",
        type=Path,
        help="日志文件或目录路径",
    )
    analyze_parser.add_argument(
        "-f",
        "--format",
        choices=["nginx", "app", "json", "auto"],
        default="auto",
        help="日志格式 (default: auto)",
    )
    analyze_parser.add_argument(
        "-t",
        "--type",
        choices=["errors", "requests", "timeline", "all"],
        default="all",
        help="分析类型 (default: all)",
    )
    analyze_parser.add_argument(
        "--pattern",
        default="*.log",
        help="文件匹配模式 (default: *.log)",
    )
    analyze_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="显示详细输出",
    )

    # =========================================================================
    # report 命令
    # =========================================================================
    report_parser = subparsers.add_parser(
        "report",
        help="生成分析报告",
    )
    report_parser.add_argument(
        "path",
        type=Path,
        help="日志文件或目录路径",
    )
    report_parser.add_argument(
        "-f",
        "--format",
        choices=["terminal", "json", "markdown"],
        default="terminal",
        help="报告格式 (default: terminal)",
    )
    report_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出文件路径",
    )
    report_parser.add_argument(
        "--log-format",
        choices=["nginx", "app", "json", "auto"],
        default="auto",
        help="日志格式 (default: auto)",
    )
    report_parser.add_argument(
        "--no-colors",
        action="store_true",
        help="禁用彩色输出",
    )

    # =========================================================================
    # clean 命令
    # =========================================================================
    clean_parser = subparsers.add_parser(
        "clean",
        help="清理/归档日志文件",
    )
    clean_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="日志目录路径",
    )
    clean_parser.add_argument(
        "--older-than",
        type=int,
        default=30,
        metavar="DAYS",
        help="清理超过指定天数的文件 (default: 30)",
    )
    clean_parser.add_argument(
        "--archive",
        action="store_true",
        help="归档旧日志",
    )
    clean_parser.add_argument(
        "--compress",
        action="store_true",
        help="压缩日志文件",
    )
    clean_parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("./archive"),
        help="归档目录 (default: ./archive)",
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不实际执行",
    )
    clean_parser.add_argument(
        "--execute",
        action="store_true",
        help="实际执行清理",
    )
    clean_parser.add_argument(
        "--resume",
        action="store_true",
        help="恢复中断的清理任务",
    )
    clean_parser.add_argument(
        "--status",
        action="store_true",
        help="查看当前清理任务状态",
    )
    clean_parser.add_argument(
        "--pattern",
        default="*.log",
        help="文件匹配模式 (default: *.log)",
    )

    return parser


def get_log_files(path: Path, pattern: str = "*.log") -> list[Path]:
    """获取日志文件列表"""
    if path.is_file():
        return [path]
    elif path.is_dir():
        return list(path.glob(pattern))
    else:
        return []


def detect_log_format(files: list[Path]) -> str:
    """自动检测日志格式"""
    for file_path in files[:3]:  # 只检查前3个文件
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        fmt = detect_format(line)
                        if fmt:
                            return fmt
        except Exception:
            continue
    return "app"  # 默认格式


def cmd_analyze(args: argparse.Namespace) -> int:
    """执行 analyze 命令"""
    files = get_log_files(args.path, args.pattern)
    if not files:
        print(f"No log files found in {args.path}")
        return 1

    # 确定日志格式
    log_format = args.format
    if log_format == "auto":
        log_format = detect_log_format(files)
        if args.verbose:
            print(f"Detected log format: {log_format}")

    # 解析日志
    parser = get_parser(log_format)
    entries = list(parser.parse_files(files))

    if args.verbose:
        print(f"Parsed {len(entries)} entries from {len(files)} files")
        if parser.errors:
            print(f"Parse errors: {len(parser.errors)}")

    # 分析
    error_analyzer = ErrorAnalyzer()
    request_analyzer = RequestAnalyzer()
    timeline_analyzer = TimelineAnalyzer()

    for entry in entries:
        if args.type in ("errors", "all"):
            error_analyzer.analyze_entry(entry)
        if args.type in ("requests", "all"):
            request_analyzer.analyze_entry(entry)
        if args.type in ("timeline", "all"):
            timeline_analyzer.analyze_entry(entry)

    # 输出结果
    settings = get_settings()
    reporter = TerminalReporter(use_colors=settings.use_colors)

    report = AnalysisReport(
        files_analyzed=len(files),
        total_entries=len(entries),
        valid_entries=len(entries),
        invalid_entries=len(parser.errors),
        error_stats=error_analyzer.get_stats(),
        request_stats=request_analyzer.get_stats() if any(isinstance(e, NginxLogEntry) for e in entries) else None,
        timeline_stats=timeline_analyzer.get_stats(),
    )

    reporter.print(report)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """执行 report 命令"""
    files = get_log_files(args.path)
    if not files:
        print(f"No log files found in {args.path}")
        return 1

    # 确定日志格式
    log_format = args.log_format
    if log_format == "auto":
        log_format = detect_log_format(files)

    # 解析日志
    parser = get_parser(log_format)
    entries = list(parser.parse_files(files))

    # 分析
    error_analyzer = ErrorAnalyzer()
    request_analyzer = RequestAnalyzer()
    timeline_analyzer = TimelineAnalyzer()

    for entry in entries:
        error_analyzer.analyze_entry(entry)
        request_analyzer.analyze_entry(entry)
        timeline_analyzer.analyze_entry(entry)

    # 生成报告
    report = AnalysisReport(
        files_analyzed=len(files),
        total_entries=len(entries),
        valid_entries=len(entries),
        invalid_entries=len(parser.errors),
        error_stats=error_analyzer.get_stats(),
        request_stats=request_analyzer.get_stats() if any(isinstance(e, NginxLogEntry) for e in entries) else None,
        timeline_stats=timeline_analyzer.get_stats(),
    )

    # 输出报告
    report_format: Literal["terminal", "json", "markdown"] = args.format

    if report_format == "terminal":
        reporter = TerminalReporter(use_colors=not args.no_colors)
        reporter.print(report)
    elif report_format == "json":
        reporter_json = JsonReporter()
        if args.output:
            reporter_json.save(report, args.output)
            print(f"Report saved to {args.output}")
        else:
            print(reporter_json.generate(report))
    elif report_format == "markdown":
        reporter_md = MarkdownReporter()
        if args.output:
            reporter_md.save(report, args.output)
            print(f"Report saved to {args.output}")
        else:
            print(reporter_md.generate(report))

    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """执行 clean 命令"""
    settings = get_settings()

    # 查看状态
    if args.status:
        archiver = LogArchiver(settings.archive_dir, settings.state_file)
        print(archiver.get_status())
        return 0

    # 恢复任务
    if args.resume:
        archiver = LogArchiver(settings.archive_dir, settings.state_file)
        result = archiver.execute(resume=True)
        print(f"Resumed: {result.processed_files} files processed")
        if result.errors:
            for error in result.errors:
                print(f"  Error: {error}")
        return 0 if not result.errors else 1

    # 需要路径
    if not args.path:
        print("Error: path is required for clean operation")
        return 1

    # 执行清理
    if args.compress:
        compressor = LogCompressor(compress_level=settings.compress_level)
        compressor.plan(args.path, args.older_than, args.pattern)

        if args.dry_run:
            print(compressor.preview())
            return 0

        if not args.execute:
            print("Use --execute to perform the operation, or --dry-run to preview")
            return 0

        result = compressor.execute()
        print(f"Compressed: {result.processed_files} files")
        print(f"Space freed: {result.bytes_freed / (1024*1024):.2f} MB")

    else:
        # 默认归档
        archiver = LogArchiver(args.archive_dir, settings.state_file)
        archiver.plan(args.path, args.older_than, args.pattern)

        if args.dry_run:
            print(archiver.preview())
            return 0

        if not args.execute:
            print("Use --execute to perform the operation, or --dry-run to preview")
            return 0

        result = archiver.execute()
        print(f"Archived: {result.processed_files} files")
        print(f"Total size: {result.bytes_archived / (1024*1024):.2f} MB")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  {error}")
        return 1

    return 0


def main() -> int:
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "analyze": cmd_analyze,
        "report": cmd_report,
        "clean": cmd_clean,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

