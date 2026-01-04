#!/usr/bin/env python3
"""
文本统计器 - P01 综合项目

功能：
- 统计文件的行数
- 统计单词数
- 统计字符数
- 找出最长的行

用法：
    python main.py <filename>
    python main.py sample.txt
"""

import sys
from pathlib import Path


def analyze_text(content: str) -> dict:
    """
    分析文本内容

    Args:
        content: 文本内容

    Returns:
        包含统计信息的字典
    """
    lines = content.splitlines()

    # 统计行数
    line_count = len(lines)

    # 统计单词数
    word_count = sum(len(line.split()) for line in lines)

    # 统计字符数（不含换行符）
    char_count = sum(len(line) for line in lines)

    # 找最长行
    if lines:
        longest_line = max(lines, key=len)
        longest_line_length = len(longest_line)
    else:
        longest_line = ""
        longest_line_length = 0

    return {
        "lines": line_count,
        "words": word_count,
        "characters": char_count,
        "longest_line_length": longest_line_length,
        "longest_line": longest_line,
    }


def analyze_file(filepath: str) -> dict:
    """
    分析文件

    Args:
        filepath: 文件路径

    Returns:
        包含统计信息的字典

    Raises:
        FileNotFoundError: 文件不存在
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if not path.is_file():
        raise ValueError(f"不是文件: {filepath}")

    content = path.read_text(encoding="utf-8")
    return analyze_text(content)


def print_results(stats: dict, filepath: str = "") -> None:
    """打印统计结果"""
    if filepath:
        print(f"\n📄 文件: {filepath}")
        print("=" * 40)

    print(f"📊 统计结果:")
    print(f"   行数:     {stats['lines']}")
    print(f"   单词数:   {stats['words']}")
    print(f"   字符数:   {stats['characters']}")
    print(f"   最长行:   {stats['longest_line_length']} 个字符")

    if stats["longest_line"]:
        preview = stats["longest_line"][:50]
        if len(stats["longest_line"]) > 50:
            preview += "..."
        print(f"   内容预览: {preview}")


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python main.py <filename>")
        print("示例: python main.py sample.txt")

        # 如果没有参数，创建并分析示例文件
        print("\n创建示例文件并分析...")
        sample_content = """Hello, Python!
This is a sample text file.
It contains multiple lines.
The quick brown fox jumps over the lazy dog.
Python is a great programming language."""

        stats = analyze_text(sample_content)
        print_results(stats, "示例文本")
        return

    filepath = sys.argv[1]

    try:
        stats = analyze_file(filepath)
        print_results(stats, filepath)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

