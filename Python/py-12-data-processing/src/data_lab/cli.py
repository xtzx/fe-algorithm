"""命令行接口

提供数据清洗管道的 CLI。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from data_lab.cleaners import create_user_cleaner
from data_lab.models import User
from data_lab.parsers import read_csv, write_jsonl
from data_lab.reporters import generate_report
from data_lab.validators import validate_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="data-lab",
        description="数据处理与清洗工具",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
    )

    # clean 命令
    clean_parser = subparsers.add_parser(
        "clean",
        help="清洗数据文件",
    )
    clean_parser.add_argument(
        "input",
        type=Path,
        help="输入文件 (CSV)",
    )
    clean_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出文件 (JSONL)",
    )
    clean_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="报告文件 (JSON)",
    )

    # report 命令
    report_parser = subparsers.add_parser(
        "report",
        help="生成数据质量报告",
    )
    report_parser.add_argument(
        "input",
        type=Path,
        help="输入文件 (CSV/JSONL)",
    )
    report_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="报告输出文件",
    )
    report_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="报告格式",
    )

    # validate 命令
    validate_parser = subparsers.add_parser(
        "validate",
        help="验证数据",
    )
    validate_parser.add_argument(
        "input",
        type=Path,
        help="输入文件",
    )
    validate_parser.add_argument(
        "--schema",
        type=Path,
        help="Schema 文件 (JSON)",
    )

    return parser


def cmd_clean(args: argparse.Namespace) -> int:
    """执行 clean 命令"""
    input_path = args.input

    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        return 1

    logger.info(f"读取文件: {input_path}")

    # 读取数据
    try:
        data = read_csv(input_path)
    except Exception as e:
        logger.error(f"读取失败: {e}")
        return 1

    logger.info(f"读取 {len(data)} 条记录")

    # 清洗
    cleaner = create_user_cleaner()
    cleaned, stats = cleaner.clean_batch(data)

    logger.info(f"清洗成功: {stats.success}/{stats.total}")

    # 验证
    result = validate_batch(cleaned, User)

    logger.info(f"验证成功: {result.success}/{result.total}")

    # 输出
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".clean.jsonl")

    count = write_jsonl(output_path, result.cleaned_data)
    logger.info(f"写入 {count} 条记录到: {output_path}")

    # 报告
    if args.report:
        report = {
            "input": str(input_path),
            "output": str(output_path),
            "total": len(data),
            "cleaned": stats.success,
            "validated": result.success,
            "errors": result.errors[:20],  # 最多 20 个错误
        }
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"报告已保存到: {args.report}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """执行 report 命令"""
    input_path = args.input

    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        return 1

    logger.info(f"分析文件: {input_path}")

    # 读取数据
    try:
        if input_path.suffix == ".csv":
            data = read_csv(input_path)
        else:
            from data_lab.parsers import read_jsonl

            data = list(read_jsonl(input_path))
    except Exception as e:
        logger.error(f"读取失败: {e}")
        return 1

    # 生成报告
    report = generate_report(data)

    # 输出
    if args.format == "markdown":
        output = report.to_markdown()
    else:
        output = report.to_json()

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        logger.info(f"报告已保存到: {args.output}")
    else:
        print(output)

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """执行 validate 命令"""
    input_path = args.input

    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        return 1

    # 读取数据
    try:
        if input_path.suffix == ".csv":
            data = read_csv(input_path)
        else:
            from data_lab.parsers import read_jsonl

            data = list(read_jsonl(input_path))
    except Exception as e:
        logger.error(f"读取失败: {e}")
        return 1

    # 验证
    result = validate_batch(data, User)

    print(f"总计: {result.total}")
    print(f"成功: {result.success}")
    print(f"失败: {result.failed}")
    print(f"成功率: {result.success_rate}%")

    if result.errors:
        print("\n错误示例:")
        for error in result.errors[:5]:
            print(f"  Row {error['index']}: {error['errors']}")

    return 0 if result.failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """主入口"""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "clean":
        return cmd_clean(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "validate":
        return cmd_validate(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

