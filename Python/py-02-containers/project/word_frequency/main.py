#!/usr/bin/env python3
"""
词频统计器 - P02 综合项目

功能：
- 读取文本文件
- 统计每个单词出现次数
- 输出 Top N 高频词

用法：
    python main.py <filename> [--top N]
    python main.py sample.txt --top 10
"""

import sys
import re
from collections import Counter
from pathlib import Path


def read_file(filepath: str) -> str:
    """读取文件内容"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    return path.read_text(encoding="utf-8")


def extract_words(text: str) -> list[str]:
    """提取单词列表"""
    # 转小写，提取字母数字单词
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def count_words(words: list[str]) -> Counter:
    """统计词频"""
    return Counter(words)


def get_top_n(counter: Counter, n: int) -> list[tuple[str, int]]:
    """获取 Top N"""
    return counter.most_common(n)


def print_results(top_words: list[tuple[str, int]], total: int) -> None:
    """打印结果"""
    print(f"\n📊 词频统计结果")
    print(f"{'='*40}")
    print(f"总单词数: {total}")
    print(f"不同单词数: {len(top_words) if len(top_words) < total else '更多'}")
    print(f"\n{'排名':<6}{'单词':<15}{'次数':<10}{'占比':<10}")
    print(f"{'-'*40}")

    for i, (word, count) in enumerate(top_words, 1):
        percentage = count / total * 100
        print(f"{i:<6}{word:<15}{count:<10}{percentage:.2f}%")


def analyze_text(text: str, top_n: int = 10) -> dict:
    """分析文本"""
    words = extract_words(text)
    counter = count_words(words)
    top_words = get_top_n(counter, top_n)

    return {
        "total_words": len(words),
        "unique_words": len(counter),
        "top_words": top_words,
        "counter": counter,
    }


def main():
    """主函数"""
    # 解析参数
    if len(sys.argv) < 2:
        print("用法: python main.py <filename> [--top N]")
        print("示例: python main.py sample.txt --top 10")
        print("\n运行示例文本...")

        sample_text = """
        Python is a great programming language.
        Python is easy to learn and use.
        Many developers love Python for its simplicity.
        Python can be used for web development, data science, and automation.
        Learning Python is a great investment for your career.
        """

        result = analyze_text(sample_text, 10)
        print_results(result["top_words"], result["total_words"])
        return

    filepath = sys.argv[1]
    top_n = 10

    # 解析 --top 参数
    if "--top" in sys.argv:
        try:
            idx = sys.argv.index("--top")
            top_n = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("错误: --top 参数需要一个整数")
            sys.exit(1)

    try:
        text = read_file(filepath)
        result = analyze_text(text, top_n)
        print(f"\n📄 文件: {filepath}")
        print_results(result["top_words"], result["total_words"])
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

