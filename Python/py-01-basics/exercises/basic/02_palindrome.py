#!/usr/bin/env python3
"""
练习 2：回文判断

题目：判断一个字符串是否是回文（忽略大小写和空格）

示例：
    "A man a plan a canal Panama" -> True
    "hello" -> False
"""


def is_palindrome(s: str) -> bool:
    """判断字符串是否是回文"""
    # TODO: 实现回文判断逻辑
    pass


# 测试
if __name__ == "__main__":
    test_cases = [
        ("A man a plan a canal Panama", True),
        ("hello", False),
        ("racecar", True),
        ("Was it a car or a cat I saw", True),
        ("", True),
        ("a", True),
    ]

    for s, expected in test_cases:
        result = is_palindrome(s)
        status = "✅" if result == expected else "❌"
        print(f"{status} is_palindrome('{s}') = {result}, expected {expected}")

