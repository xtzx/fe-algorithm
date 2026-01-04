#!/usr/bin/env python3
"""
练习答案：词频统计
"""


def word_frequency(text: str) -> dict[str, int]:
    """统计单词频率"""
    words = text.lower().split()
    freq = {}
    for word in words:
        # 去除标点
        word = word.strip(".,!?;:")
        freq[word] = freq.get(word, 0) + 1
    return freq


# 使用 Counter 的版本
def word_frequency_v2(text: str) -> dict[str, int]:
    from collections import Counter

    words = text.lower().split()
    words = [w.strip(".,!?;:") for w in words]
    return dict(Counter(words))


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

