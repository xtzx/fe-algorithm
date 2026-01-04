#!/usr/bin/env python3
"""argparse 模块演示

使用方法:
    python 08_argparse_demo.py
    python 08_argparse_demo.py --help
    python 08_argparse_demo.py input.txt
    python 08_argparse_demo.py input.txt -o output.txt -v
    python 08_argparse_demo.py input.txt --format json --count 5
"""

import argparse


def demo_basic():
    """基本用法演示"""
    print("=" * 50)
    print("1. 基本用法")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="基本演示")

    # 位置参数
    parser.add_argument("filename", help="输入文件名")

    # 可选参数
    parser.add_argument("-o", "--output", help="输出文件名")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    # 模拟解析
    args = parser.parse_args(["input.txt", "-o", "output.txt", "-v"])

    print(f"filename: {args.filename}")
    print(f"output: {args.output}")
    print(f"verbose: {args.verbose}")


def demo_types():
    """参数类型"""
    print("\n" + "=" * 50)
    print("2. 参数类型")
    print("=" * 50)

    parser = argparse.ArgumentParser()

    # 类型转换
    parser.add_argument("-n", "--number", type=int, default=10, help="数量")
    parser.add_argument("-f", "--factor", type=float, default=1.0, help="因子")

    # 多个值
    parser.add_argument("--items", nargs="+", help="一个或多个项目")
    parser.add_argument("--pair", nargs=2, help="恰好两个值")

    # 选择
    parser.add_argument("--level", choices=["debug", "info", "error"], default="info")

    # 模拟解析
    args = parser.parse_args([
        "-n", "20",
        "-f", "1.5",
        "--items", "a", "b", "c",
        "--pair", "x", "y",
        "--level", "debug"
    ])

    print(f"number: {args.number} (type: {type(args.number).__name__})")
    print(f"factor: {args.factor} (type: {type(args.factor).__name__})")
    print(f"items: {args.items}")
    print(f"pair: {args.pair}")
    print(f"level: {args.level}")


def demo_actions():
    """action 参数"""
    print("\n" + "=" * 50)
    print("3. action 参数")
    print("=" * 50)

    parser = argparse.ArgumentParser()

    # store_true/store_false
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_false", dest="show_output")

    # count
    parser.add_argument("-d", "--debug", action="count", default=0)

    # append
    parser.add_argument("-i", "--include", action="append", default=[])

    # 模拟解析
    args = parser.parse_args(["-v", "-ddd", "-i", "a", "-i", "b"])

    print(f"verbose: {args.verbose}")
    print(f"show_output: {args.show_output}")
    print(f"debug: {args.debug} (使用 -ddd)")
    print(f"include: {args.include}")


def demo_groups():
    """参数组"""
    print("\n" + "=" * 50)
    print("4. 参数组")
    print("=" * 50)

    parser = argparse.ArgumentParser()

    # 分组
    input_group = parser.add_argument_group("输入选项")
    input_group.add_argument("-i", "--input", help="输入文件")
    input_group.add_argument("--encoding", default="utf-8", help="编码")

    output_group = parser.add_argument_group("输出选项")
    output_group.add_argument("-o", "--output", help="输出文件")
    output_group.add_argument("--format", choices=["json", "csv"], default="json")

    # 互斥组
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-v", "--verbose", action="store_true")
    mode_group.add_argument("-q", "--quiet", action="store_true")

    # 打印帮助（演示分组效果）
    print("帮助信息:")
    parser.print_help()


def demo_subcommands():
    """子命令"""
    print("\n" + "=" * 50)
    print("5. 子命令")
    print("=" * 50)

    parser = argparse.ArgumentParser(prog="git-like")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # add 子命令
    add_parser = subparsers.add_parser("add", help="添加文件")
    add_parser.add_argument("files", nargs="+", help="要添加的文件")

    # commit 子命令
    commit_parser = subparsers.add_parser("commit", help="提交更改")
    commit_parser.add_argument("-m", "--message", required=True, help="提交信息")

    # push 子命令
    push_parser = subparsers.add_parser("push", help="推送到远程")
    push_parser.add_argument("--force", action="store_true", help="强制推送")

    # 模拟解析
    print("解析: add file1.py file2.py")
    args = parser.parse_args(["add", "file1.py", "file2.py"])
    print(f"  command: {args.command}")
    print(f"  files: {args.files}")

    print("\n解析: commit -m 'fix bug'")
    args = parser.parse_args(["commit", "-m", "fix bug"])
    print(f"  command: {args.command}")
    print(f"  message: {args.message}")

    print("\n解析: push --force")
    args = parser.parse_args(["push", "--force"])
    print(f"  command: {args.command}")
    print(f"  force: {args.force}")


def demo_complete_example():
    """完整示例"""
    print("\n" + "=" * 50)
    print("6. 完整示例")
    print("=" * 50)

    parser = argparse.ArgumentParser(
        description="文件处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.txt -o output.txt
  %(prog)s input.txt --format json -v
        """
    )

    # 位置参数
    parser.add_argument("input", help="输入文件")

    # 可选参数
    parser.add_argument("-o", "--output", help="输出文件（默认：stdout）")
    parser.add_argument("-f", "--format", choices=["text", "json", "csv"],
                        default="text", help="输出格式")
    parser.add_argument("-n", "--count", type=int, default=10, help="处理数量")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    # 模拟解析
    args = parser.parse_args(["data.txt", "-o", "result.json", "-f", "json", "-v", "-n", "100"])

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"format: {args.format}")
    print(f"count: {args.count}")
    print(f"verbose: {args.verbose}")


if __name__ == "__main__":
    demo_basic()
    demo_types()
    demo_actions()
    demo_groups()
    demo_subcommands()
    demo_complete_example()

    print("\n✅ argparse 演示完成!")


