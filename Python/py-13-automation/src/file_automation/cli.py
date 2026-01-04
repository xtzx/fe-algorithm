"""
命令行接口

file-auto 命令行工具
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from file_automation.operations import Operation
from file_automation.planner import (
    RenamePlanner,
    OrganizePlanner,
    CleanupPlanner,
    analyze_plan,
)
from file_automation.executor import Executor, preview_operations, confirm_execution
from file_automation.state import StateManager


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def cmd_rename(args: argparse.Namespace) -> int:
    """重命名命令"""
    directory = Path(args.directory)

    if not directory.exists():
        print(f"错误：目录不存在: {directory}")
        return 1

    planner = RenamePlanner(directory)

    # 根据模式创建计划
    if args.mode == "regex":
        if not args.pattern or not args.replacement:
            print("错误：正则模式需要 --pattern 和 --replacement")
            return 1
        operations = planner.plan_regex_rename(
            pattern=args.pattern,
            replacement=args.replacement,
            file_glob=args.glob,
        )
    elif args.mode == "sequential":
        if not args.prefix:
            print("错误：序号模式需要 --prefix")
            return 1
        operations = planner.plan_sequential_rename(
            prefix=args.prefix,
            file_glob=args.glob,
            start=args.start or 1,
            width=args.width or 3,
            sort_by=args.sort or "name",
        )
    elif args.mode == "datetime":
        operations = planner.plan_datetime_rename(
            file_glob=args.glob,
            date_format=args.date_format or "%Y%m%d_%H%M%S",
        )
    else:
        print(f"错误：未知模式: {args.mode}")
        return 1

    return _execute_operations(operations, args)


def cmd_organize(args: argparse.Namespace) -> int:
    """整理命令"""
    source = Path(args.source)
    target = Path(args.target) if args.target else None

    if not source.exists():
        print(f"错误：目录不存在: {source}")
        return 1

    planner = OrganizePlanner(source, target)

    # 根据分类方式创建计划
    if args.by == "extension":
        operations = planner.plan_by_extension()
    elif args.by == "date":
        operations = planner.plan_by_date(
            date_format=args.date_format or "%Y/%Y-%m",
        )
    elif args.by == "size":
        operations = planner.plan_by_size()
    else:
        print(f"错误：未知分类方式: {args.by}")
        return 1

    return _execute_operations(operations, args)


def cmd_clean(args: argparse.Namespace) -> int:
    """清理命令"""
    directory = Path(args.directory)

    if not directory.exists():
        print(f"错误：目录不存在: {directory}")
        return 1

    planner = CleanupPlanner(directory)

    if args.empty_dirs:
        operations = planner.plan_delete_empty_dirs()
    elif args.patterns:
        operations = planner.plan_delete_by_pattern(args.patterns)
    elif args.older_than:
        operations = planner.plan_delete_old_files(
            max_age_days=args.older_than,
            file_glob=args.glob,
        )
    else:
        print("错误：请指定清理条件 (--empty-dirs, --patterns, --older-than)")
        return 1

    return _execute_operations(operations, args)


def cmd_status(args: argparse.Namespace) -> int:
    """查看状态"""
    state_file = Path(args.state_file)

    if not state_file.exists():
        print("没有进行中的批处理")
        return 0

    manager = StateManager(state_file)
    state = manager.load()

    if state is None:
        print("状态文件损坏")
        return 1

    print(f"批次 ID: {state.batch_id}")
    print(f"创建时间: {state.created_at}")
    print(f"更新时间: {state.updated_at}")
    print()
    print(f"总任务数: {state.total_tasks}")
    print(f"  已完成: {state.completed_count}")
    print(f"  失败: {state.failed_count}")
    print(f"  待执行: {state.pending_count}")
    print(f"  跳过: {state.skipped_count}")
    print(f"  进度: {state.progress:.1f}%")

    if args.show_failed and state.failed_count > 0:
        print()
        print("失败的任务:")
        for idx, task in state.tasks.items():
            if task.status.value == "failed":
                print(f"  [{idx}] {task.error}")

    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """恢复执行"""
    state_file = Path(args.state_file)

    if not state_file.exists():
        print("没有可恢复的批处理")
        return 1

    manager = StateManager(state_file)
    state = manager.load()

    if state is None:
        print("状态文件损坏")
        return 1

    if state.is_complete:
        print("批处理已完成")
        return 0

    pending = manager.get_pending_indices()
    print(f"将恢复执行 {len(pending)} 个待处理任务")

    if not args.execute:
        print("使用 --execute 实际执行")
        return 0

    # 需要重新加载操作计划...
    # 这里简化处理，实际应该保存计划
    print("注意：恢复功能需要配合保存的操作计划使用")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """回滚操作"""
    rollback_file = Path(args.rollback_file)

    if not rollback_file.exists():
        print("没有可回滚的操作")
        return 0

    executor = Executor(
        dry_run=not args.execute,
        rollback_file=rollback_file,
    )

    count = args.count if args.count else None

    if not args.execute:
        print("预览模式，使用 --execute 实际回滚")

    rolled_back = executor.rollback(count)
    print(f"回滚了 {rolled_back} 个操作")

    return 0


def _execute_operations(operations: list[Operation], args: argparse.Namespace) -> int:
    """执行操作列表"""
    if not operations:
        print("没有需要执行的操作")
        return 0

    # 显示预览
    print(preview_operations(operations))

    # 分析计划
    analysis = analyze_plan(operations)
    print(f"\n操作统计: {analysis['by_type']}")

    # Dry-run 模式
    if not args.execute:
        print("\n这是预览模式，使用 --execute 或 -x 实际执行")
        return 0

    # 确认执行
    if not args.yes and not confirm_execution():
        print("已取消")
        return 0

    # 配置执行器
    state_file = Path(args.state_file) if hasattr(args, "state_file") else None
    rollback_file = Path(args.rollback_file) if hasattr(args, "rollback_file") else None

    executor = Executor(
        dry_run=False,
        state_file=state_file,
        rollback_file=rollback_file,
        max_retries=args.retries if hasattr(args, "retries") else 3,
        continue_on_error=not args.stop_on_error
        if hasattr(args, "stop_on_error")
        else True,
    )

    # 执行
    summary = executor.execute(operations)

    # 输出结果
    print()
    print("=" * 60)
    print("执行结果")
    print("=" * 60)
    print(f"总计: {summary.total}")
    print(f"成功: {summary.success}")
    print(f"失败: {summary.failed}")
    print(f"跳过: {summary.skipped}")
    print(f"成功率: {summary.success_rate:.1f}%")
    print(f"耗时: {summary.duration_ms:.0f}ms")

    return 0 if summary.failed == 0 else 1


def main() -> int:
    """主入口"""
    parser = argparse.ArgumentParser(
        prog="file-auto",
        description="文件批处理自动化工具",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="日志文件路径",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # rename 命令
    rename_parser = subparsers.add_parser("rename", help="批量重命名")
    rename_parser.add_argument("directory", type=str, help="目标目录")
    rename_parser.add_argument(
        "-m",
        "--mode",
        choices=["regex", "sequential", "datetime"],
        default="regex",
        help="重命名模式",
    )
    rename_parser.add_argument("-p", "--pattern", help="正则表达式")
    rename_parser.add_argument("-r", "--replacement", help="替换字符串")
    rename_parser.add_argument("--prefix", help="序号前缀")
    rename_parser.add_argument("--start", type=int, help="起始序号")
    rename_parser.add_argument("--width", type=int, help="序号宽度")
    rename_parser.add_argument("--sort", choices=["name", "date", "size"], help="排序方式")
    rename_parser.add_argument("--date-format", help="日期格式")
    rename_parser.add_argument("-g", "--glob", default="*", help="文件过滤模式")
    rename_parser.add_argument("-x", "--execute", action="store_true", help="实际执行")
    rename_parser.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    rename_parser.add_argument("--state-file", default=".file-auto-state.json")
    rename_parser.add_argument("--rollback-file", default=".file-auto-rollback.json")
    rename_parser.add_argument("--retries", type=int, default=3)
    rename_parser.add_argument("--stop-on-error", action="store_true")
    rename_parser.set_defaults(func=cmd_rename)

    # organize 命令
    organize_parser = subparsers.add_parser("organize", help="文件整理")
    organize_parser.add_argument("source", type=str, help="源目录")
    organize_parser.add_argument("-t", "--target", help="目标目录")
    organize_parser.add_argument(
        "-b",
        "--by",
        choices=["extension", "date", "size"],
        default="extension",
        help="分类方式",
    )
    organize_parser.add_argument("--date-format", help="日期目录格式")
    organize_parser.add_argument("-x", "--execute", action="store_true", help="实际执行")
    organize_parser.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    organize_parser.add_argument("--state-file", default=".file-auto-state.json")
    organize_parser.add_argument("--rollback-file", default=".file-auto-rollback.json")
    organize_parser.set_defaults(func=cmd_organize)

    # clean 命令
    clean_parser = subparsers.add_parser("clean", help="文件清理")
    clean_parser.add_argument("directory", type=str, help="目标目录")
    clean_parser.add_argument("--empty-dirs", action="store_true", help="删除空目录")
    clean_parser.add_argument("--patterns", nargs="+", help="删除匹配的文件")
    clean_parser.add_argument("--older-than", type=int, help="删除超过 N 天的文件")
    clean_parser.add_argument("-g", "--glob", default="*", help="文件过滤模式")
    clean_parser.add_argument("-x", "--execute", action="store_true", help="实际执行")
    clean_parser.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    clean_parser.add_argument("--state-file", default=".file-auto-state.json")
    clean_parser.add_argument("--rollback-file", default=".file-auto-rollback.json")
    clean_parser.set_defaults(func=cmd_clean)

    # status 命令
    status_parser = subparsers.add_parser("status", help="查看状态")
    status_parser.add_argument("--state-file", default=".file-auto-state.json")
    status_parser.add_argument("--show-failed", action="store_true", help="显示失败详情")
    status_parser.set_defaults(func=cmd_status)

    # resume 命令
    resume_parser = subparsers.add_parser("resume", help="恢复执行")
    resume_parser.add_argument("--state-file", default=".file-auto-state.json")
    resume_parser.add_argument("-x", "--execute", action="store_true", help="实际执行")
    resume_parser.set_defaults(func=cmd_resume)

    # rollback 命令
    rollback_parser = subparsers.add_parser("rollback", help="回滚操作")
    rollback_parser.add_argument("--rollback-file", default=".file-auto-rollback.json")
    rollback_parser.add_argument("-c", "--count", type=int, help="回滚数量")
    rollback_parser.add_argument("-x", "--execute", action="store_true", help="实际执行")
    rollback_parser.set_defaults(func=cmd_rollback)

    # 解析参数
    args = parser.parse_args()

    # 配置日志
    setup_logging(args.verbose, args.log_file)

    # 执行命令
    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

