"""
CLI 命令行接口

基于 argparse 的 CLI 框架模板
"""

import argparse
import sys
from pathlib import Path

from scaffold import __version__
from scaffold.config import get_settings
from scaffold.log import get_logger, setup_logging


def cmd_run(args: argparse.Namespace) -> int:
    """
    运行主程序

    示例命令：scaffold run --config config.toml
    """
    logger = get_logger(__name__)
    settings = get_settings()

    logger.info(f"Starting {settings.app_name}...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug mode: {settings.debug}")

    if args.config:
        logger.info(f"Config file: {args.config}")

    # 这里添加你的主程序逻辑
    logger.info("Application running...")

    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """
    显示版本信息

    示例命令：scaffold version
    """
    print(f"scaffold version {__version__}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """
    显示配置信息

    示例命令：scaffold config
    """
    settings = get_settings()

    if args.json:
        print(settings.model_dump_json(indent=2))
    else:
        print("=" * 60)
        print("Current Configuration")
        print("=" * 60)
        for key, value in settings.model_dump().items():
            # 隐藏敏感信息
            if "secret" in key.lower() or "password" in key.lower():
                value = "***" if value else None
            print(f"  {key}: {value}")

    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """
    初始化项目

    示例命令：scaffold init my-project
    """
    logger = get_logger(__name__)
    project_name = args.name

    logger.info(f"Initializing project: {project_name}")

    # 创建项目目录
    project_dir = Path(project_name)
    if project_dir.exists():
        logger.error(f"Directory already exists: {project_dir}")
        return 1

    # 这里可以添加项目初始化逻辑
    # 例如：复制模板文件、创建目录结构等

    logger.info(f"Project initialized: {project_dir}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="scaffold",
        description="Python 项目工程化脚手架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run                 # 运行程序
  %(prog)s run --config app.toml
  %(prog)s version             # 显示版本
  %(prog)s config              # 显示配置
  %(prog)s config --json       # JSON 格式
  %(prog)s init my-project     # 初始化新项目
        """,
    )

    # 全局参数
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出 (DEBUG 级别)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="安静模式 (只显示错误)",
    )
    parser.add_argument(
        "--json-log",
        action="store_true",
        help="使用 JSON 格式日志",
    )

    # 子命令
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="可用命令",
    )

    # run 命令
    run_parser = subparsers.add_parser(
        "run",
        help="运行程序",
        description="运行主程序",
    )
    run_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="配置文件路径",
    )
    run_parser.set_defaults(func=cmd_run)

    # version 命令
    version_parser = subparsers.add_parser(
        "version",
        help="显示版本",
        description="显示版本信息",
    )
    version_parser.set_defaults(func=cmd_version)

    # config 命令
    config_parser = subparsers.add_parser(
        "config",
        help="显示配置",
        description="显示当前配置",
    )
    config_parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 格式输出",
    )
    config_parser.set_defaults(func=cmd_config)

    # init 命令
    init_parser = subparsers.add_parser(
        "init",
        help="初始化项目",
        description="初始化新项目",
    )
    init_parser.add_argument(
        "name",
        help="项目名称",
    )
    init_parser.set_defaults(func=cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI 主入口

    Args:
        argv: 命令行参数（默认使用 sys.argv）

    Returns:
        退出码
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # 配置日志
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    else:
        log_level = get_settings().log_level

    log_format = "json" if args.json_log else "text"
    setup_logging(level=log_level, format=log_format)

    # 没有子命令时显示帮助
    if args.command is None:
        parser.print_help()
        return 0

    # 执行子命令
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        logger = get_logger(__name__)
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

