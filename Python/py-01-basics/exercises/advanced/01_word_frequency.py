#!/usr/bin/env python3
"""
练习：词频统计

题目：统计字符串中每个单词出现的次数

示例：
    "Hello world. Hello Python. Python is great!"
    -> {'hello': 2, 'world': 1, 'python': 2, 'is': 1, 'great': 1}
"""


def word_frequency(text: str) -> dict[str, int]:
    """统计单词频率"""
    # TODO: 实现词频统计逻辑
    pass


# 测试
if __name__ == "__main__":
    text = "Hello world. Hello Python. Python is great!"
    result = word_frequency(text)
    print(f"输入: {text}")
    print(f"结果: {result}")

    # 验证
    assert result.get("hello") == 2, "hello 应该出现 2 次"
    assert result.get("python") == 2, "python 应该出现 2 次"
    assert result.get("world") == 1, "world 应该出现 1 次"
    print("✅ 测试通过")

