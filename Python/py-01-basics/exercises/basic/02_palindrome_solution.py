#!/usr/bin/env python3
"""
练习 2 答案：回文判断
"""


def is_palindrome(s: str) -> bool:
    """判断字符串是否是回文"""
    # 去掉空格，转小写
    s = s.replace(" ", "").lower()
    return s == s[::-1]


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

