# 08. argparse - 命令行参数

## 本节目标

- 掌握 ArgumentParser 基本用法
- 学会定义位置参数和可选参数
- 了解子命令

---

## 基本使用

```python
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="我的程序说明")

# 添加参数
parser.add_argument("filename", help="输入文件名")
parser.add_argument("-o", "--output", help="输出文件名")

# 解析参数
args = parser.parse_args()

print(args.filename)
print(args.output)
```

运行：
```bash
python script.py input.txt -o output.txt
```

---

## 位置参数

必须提供，按顺序解析。

```python
import argparse

parser = argparse.ArgumentParser()

# 必填位置参数
parser.add_argument("source", help="源文件")
parser.add_argument("dest", help="目标文件")

args = parser.parse_args()
# python script.py file1.txt file2.txt
```

---

## 可选参数

以 `-` 或 `--` 开头。

```python
import argparse

parser = argparse.ArgumentParser()

# 短选项 + 长选项
parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
parser.add_argument("-n", "--number", type=int, default=10, help="数量")
parser.add_argument("-o", "--output", required=True, help="输出文件")

args = parser.parse_args()
# python script.py -v -n 20 -o out.txt
```

---

## 参数类型

```python
import argparse

parser = argparse.ArgumentParser()

# 类型转换
parser.add_argument("-n", type=int, help="整数")
parser.add_argument("-f", type=float, help="浮点数")
parser.add_argument("--file", type=argparse.FileType("r"), help="文件")

# 多个值
parser.add_argument("--items", nargs="+", help="一个或多个")
parser.add_argument("--pair", nargs=2, help="恰好两个")
parser.add_argument("--optional", nargs="?", help="零个或一个")
parser.add_argument("--all", nargs="*", help="零个或多个")

# 选择
parser.add_argument("--level", choices=["debug", "info", "error"], help="日志级别")

args = parser.parse_args()
```

---

## action 参数

```python
import argparse

parser = argparse.ArgumentParser()

# store_true/store_false - 布尔标志
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-q", "--quiet", action="store_false")

# count - 计数
parser.add_argument("-d", "--debug", action="count", default=0)
# -ddd 结果为 3

# append - 累加
parser.add_argument("-i", "--include", action="append")
# -i a -i b 结果为 ['a', 'b']

# store_const - 存储常量
parser.add_argument("--json", action="store_const", const="json", dest="format")

args = parser.parse_args()
```

---

## 默认值

```python
import argparse

parser = argparse.ArgumentParser()

# 显式默认值
parser.add_argument("-n", "--number", type=int, default=10)

# 从环境变量
import os
parser.add_argument("--host", default=os.getenv("HOST", "localhost"))

args = parser.parse_args()
```

---

## 参数组

```python
import argparse

parser = argparse.ArgumentParser()

# 分组
input_group = parser.add_argument_group("输入选项")
input_group.add_argument("-i", "--input", help="输入文件")

output_group = parser.add_argument_group("输出选项")
output_group.add_argument("-o", "--output", help="输出文件")

# 互斥组
exclusive_group = parser.add_mutually_exclusive_group()
exclusive_group.add_argument("-v", "--verbose", action="store_true")
exclusive_group.add_argument("-q", "--quiet", action="store_true")

args = parser.parse_args()
```

---

## 子命令

```python
import argparse

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

args = parser.parse_args()

# 处理子命令
if args.command == "add":
    print(f"添加文件: {args.files}")
elif args.command == "commit":
    print(f"提交: {args.message}")
elif args.command == "push":
    print(f"推送 (强制: {args.force})")
```

运行：
```bash
python script.py add file1.py file2.py
python script.py commit -m "fix bug"
python script.py push --force
```

---

## 完整示例

```python
#!/usr/bin/env python3
import argparse
import sys

def main():
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
    parser.add_argument(
        "-o", "--output",
        help="输出文件（默认：stdout）"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="输出格式"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"输入: {args.input}")
        print(f"输出: {args.output or 'stdout'}")
        print(f"格式: {args.format}")

    # 处理逻辑...

if __name__ == "__main__":
    main()
```

---

## vs click

argparse 是标准库，click 是第三方库，更易用：

```python
# argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True)
args = parser.parse_args()

# click
import click

@click.command()
@click.option("-n", "--name", required=True)
def main(name):
    print(f"Hello {name}")

main()
```

---

## 本节要点

1. `ArgumentParser()` 创建解析器
2. `add_argument()` 添加参数
3. 位置参数必填，可选参数以 `-` 开头
4. `type`、`default`、`choices`、`help` 常用选项
5. `action="store_true"` 布尔标志
6. `add_subparsers()` 创建子命令


