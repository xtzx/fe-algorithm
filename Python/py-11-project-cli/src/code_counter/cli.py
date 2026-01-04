"""命令行接口

使用 argparse 实现 CLI。
"""

import argparse
import logging
import sys
from pathlib import Path

from code_counter import __version__
from code_counter.config import Config, init_config, show_config
from code_counter.counter import count_directory
from code_counter.output import OutputFormat, format_output

# 配置日志
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="code-counter",
        description="代码统计工具 - 统计目录下的代码行数",
        epilog="示例: code-counter scan . --format json",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="详细输出 (-v: INFO, -vv: DEBUG)",
    )

    # 子命令
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="可用命令",
    )

    # scan 命令
    scan_parser = subparsers.add_parser(
        "scan",
        help="扫描并统计代码",
        description="递归扫描目录，统计代码行数",
    )
    scan_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="要扫描的目录 (默认: 当前目录)",
    )
    scan_parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="排除的文件/目录模式 (可多次使用)",
    )
    scan_parser.add_argument(
        "-f",
        "--format",
        choices=["table", "json", "markdown"],
        default=None,
        help="输出格式 (默认: table)",
    )
    scan_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="输出到文件",
    )
    scan_parser.add_argument(
        "--no-ignore",
        action="store_true",
        help="不读取 .gitignore",
    )
    scan_parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="包含隐藏文件",
    )

    # report 命令
    report_parser = subparsers.add_parser(
        "report",
        help="生成详细报告",
        description="生成代码统计的详细报告",
    )
    report_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="要分析的目录",
    )
    report_parser.add_argument(
        "-f",
        "--format",
        choices=["table", "json", "markdown"],
        default="markdown",
        help="报告格式 (默认: markdown)",
    )
    report_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="输出到文件",
    )

    # config 命令
    config_parser = subparsers.add_parser(
        "config",
        help="配置管理",
        description="查看和管理配置",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        title="config commands",
    )

    # config show
    config_subparsers.add_parser(
        "show",
        help="显示当前配置",
    )

    # config init
    config_init = config_subparsers.add_parser(
        "init",
        help="初始化配置文件",
    )
    config_init.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="覆盖已存在的配置文件",
    )

    return parser


def cmd_scan(args: argparse.Namespace, config: Config) -> int:
    """执行 scan 命令"""
    path = args.path.resolve()

    if not path.exists():
        print(f"错误: 目录不存在: {path}", file=sys.stderr)
        return 1

    if not path.is_dir():
        print(f"错误: 不是目录: {path}", file=sys.stderr)
        return 1

    # 合并配置
    exclude_patterns = list(config.exclude) + args.exclude
    use_gitignore = not args.no_ignore and config.use_gitignore
    fmt: OutputFormat = args.format or config.default_format  # type: ignore

    logger.info(f"Scanning {path}")
    logger.debug(f"Exclude patterns: {exclude_patterns}")

    # 执行扫描
    try:
        result = count_directory(
            path=path,
            exclude_patterns=exclude_patterns,
            use_gitignore=use_gitignore,
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    # 格式化输出
    output = format_output(result, fmt)

    # 输出
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"结果已保存到: {args.output}")
    else:
        print(output)

    return 0


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    """执行 report 命令"""
    path = args.path.resolve()

    if not path.exists():
        print(f"错误: 目录不存在: {path}", file=sys.stderr)
        return 1

    # 扫描
    result = count_directory(
        path=path,
        exclude_patterns=config.exclude,
        use_gitignore=config.use_gitignore,
    )

    # 格式化
    fmt: OutputFormat = args.format
    output = format_output(result, fmt)

    # 输出
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"报告已保存到: {args.output}")
    else:
        print(output)

    return 0


def cmd_config(args: argparse.Namespace, config: Config) -> int:
    """执行 config 命令"""
    if args.config_command == "show":
        print(show_config(config))
        return 0

    elif args.config_command == "init":
        config_path = Path.cwd() / ".code-counter.toml"

        if config_path.exists() and not args.force:
            print(f"错误: 配置文件已存在: {config_path}", file=sys.stderr)
            print("使用 --force 覆盖", file=sys.stderr)
            return 1

        if config_path.exists():
            config_path.unlink()

        try:
            path = init_config()
            print(f"配置文件已创建: {path}")
            return 0
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1

    else:
        print("使用 'code-counter config show' 或 'code-counter config init'")
        return 1


def main(argv: list[str] | None = None) -> int:
    """主入口"""
    parser = create_parser()
    args = parser.parse_args(argv)

    # 配置日志级别
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)

    # 加载配置
    config = Config.load()

    # 执行命令
    if args.command == "scan":
        return cmd_scan(args, config)
    elif args.command == "report":
        return cmd_report(args, config)
    elif args.command == "config":
        return cmd_config(args, config)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

