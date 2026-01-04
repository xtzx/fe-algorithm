"""
ZipApp 演示应用

一个简单的 CLI 工具，演示如何打包为单文件可执行
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def cmd_hello(args):
    """打招呼命令"""
    name = args.name or "World"
    print(f"Hello, {name}!")
    if args.json:
        print(json.dumps({"greeting": f"Hello, {name}!", "timestamp": datetime.now().isoformat()}))


def cmd_info(args):
    """显示信息命令"""
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "timestamp": datetime.now().isoformat(),
    }
    
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print("System Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")


def cmd_calc(args):
    """计算命令"""
    try:
        result = eval(args.expression)  # 注意：实际应用中不要使用 eval
        if args.json:
            print(json.dumps({"expression": args.expression, "result": result}))
        else:
            print(f"{args.expression} = {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        prog="myapp",
        description="ZipApp Demo Application",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # hello 命令
    hello_parser = subparsers.add_parser("hello", help="Say hello")
    hello_parser.add_argument("--name", "-n", help="Name to greet")
    hello_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    hello_parser.set_defaults(func=cmd_hello)
    
    # info 命令
    info_parser = subparsers.add_parser("info", help="Show system info")
    info_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    info_parser.set_defaults(func=cmd_info)
    
    # calc 命令
    calc_parser = subparsers.add_parser("calc", help="Calculate expression")
    calc_parser.add_argument("expression", help="Math expression to calculate")
    calc_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    calc_parser.set_defaults(func=cmd_calc)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


