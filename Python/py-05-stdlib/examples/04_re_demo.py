#!/usr/bin/env python3
"""re 正则表达式模块演示"""

import re


def demo_basic_matching():
    """基本匹配"""
    print("=" * 50)
    print("1. 基本匹配")
    print("=" * 50)

    # match - 从开头匹配
    result = re.match(r"\d+", "123abc")
    print(f"match(r'\\d+', '123abc'): {result.group() if result else None}")

    result = re.match(r"\d+", "abc123")
    print(f"match(r'\\d+', 'abc123'): {result}")  # None

    # search - 搜索第一个匹配
    result = re.search(r"\d+", "abc123def456")
    print(f"search(r'\\d+', 'abc123def456'): {result.group() if result else None}")

    # fullmatch - 完整匹配
    result = re.fullmatch(r"\d+", "123")
    print(f"fullmatch(r'\\d+', '123'): {result.group() if result else None}")

    result = re.fullmatch(r"\d+", "123abc")
    print(f"fullmatch(r'\\d+', '123abc'): {result}")  # None


def demo_findall():
    """查找所有匹配"""
    print("\n" + "=" * 50)
    print("2. 查找所有匹配")
    print("=" * 50)

    text = "电话：13812345678，备用：13987654321"

    # findall
    numbers = re.findall(r"1\d{10}", text)
    print(f"findall 手机号: {numbers}")

    # 带分组的 findall
    matches = re.findall(r"(\d{3})(\d{8})", text)
    print(f"findall 分组: {matches}")

    # finditer
    print("finditer:")
    for match in re.finditer(r"\d+", "价格：100元，折扣：80元"):
        print(f"  匹配: {match.group()}, 位置: {match.span()}")


def demo_substitution():
    """替换"""
    print("\n" + "=" * 50)
    print("3. 替换")
    print("=" * 50)

    text = "Hello World"

    # 简单替换
    result = re.sub(r"World", "Python", text)
    print(f"sub('World', 'Python'): {result}")

    # 函数替换
    def upper_replace(match):
        return match.group().upper()

    result = re.sub(r"\b\w+\b", upper_replace, text)
    print(f"sub (函数): {result}")

    # 限制替换次数
    text = "aaa bbb ccc"
    result = re.sub(r"\w+", "X", text, count=2)
    print(f"sub (count=2): {result}")

    # subn - 返回替换次数
    result, count = re.subn(r"\w+", "X", "one two three")
    print(f"subn: {result}, 替换 {count} 次")


def demo_split():
    """分割"""
    print("\n" + "=" * 50)
    print("4. 分割")
    print("=" * 50)

    text = "one, two;  three    four"

    # 按正则分割
    parts = re.split(r"[,;\s]+", text)
    print(f"split: {parts}")

    # 保留分隔符
    parts = re.split(r"([,;])", "a,b;c")
    print(f"split (保留分隔符): {parts}")

    # 限制分割次数
    parts = re.split(r"\s+", "a b c d", maxsplit=2)
    print(f"split (maxsplit=2): {parts}")


def demo_compile():
    """编译正则"""
    print("\n" + "=" * 50)
    print("5. 编译正则")
    print("=" * 50)

    # 编译
    pattern = re.compile(r"\d+")

    result = pattern.findall("abc 123 def 456")
    print(f"findall: {result}")

    result = pattern.search("abc 123")
    print(f"search: {result.group() if result else None}")

    # 编译标志
    pattern = re.compile(r"hello", re.IGNORECASE)
    result = pattern.findall("Hello HELLO hello")
    print(f"IGNORECASE: {result}")


def demo_groups():
    """分组"""
    print("\n" + "=" * 50)
    print("6. 分组")
    print("=" * 50)

    text = "John: 25, Jane: 30"
    pattern = r"(\w+): (\d+)"

    # 基本分组
    print("基本分组:")
    for match in re.finditer(pattern, text):
        print(f"  全部: {match.group(0)}")
        print(f"  组1: {match.group(1)}")
        print(f"  组2: {match.group(2)}")
        print(f"  groups(): {match.groups()}")

    # 命名分组
    print("\n命名分组:")
    text = "2024-01-15"
    pattern = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    match = re.match(pattern, text)

    print(f"  year: {match.group('year')}")
    print(f"  month: {match.group('month')}")
    print(f"  groupdict(): {match.groupdict()}")


def demo_greedy():
    """贪婪 vs 非贪婪"""
    print("\n" + "=" * 50)
    print("7. 贪婪 vs 非贪婪")
    print("=" * 50)

    text = "<div>content</div>"

    # 贪婪匹配（默认）
    pattern = r"<.*>"
    match = re.search(pattern, text)
    print(f"贪婪 <.*>: {match.group()}")

    # 非贪婪匹配
    pattern = r"<.*?>"
    match = re.search(pattern, text)
    print(f"非贪婪 <.*?>: {match.group()}")


def demo_common_patterns():
    """常用模式"""
    print("\n" + "=" * 50)
    print("8. 常用模式")
    print("=" * 50)

    # 邮箱
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    text = "联系：user@example.com 或 test@domain.org"
    emails = re.findall(email_pattern, text)
    print(f"邮箱: {emails}")

    # URL
    url_pattern = r"https?://[^\s]+"
    text = "访问 https://example.com 或 http://test.org/path"
    urls = re.findall(url_pattern, text)
    print(f"URL: {urls}")

    # 中国手机号
    phone_pattern = r"1[3-9]\d{9}"
    text = "电话：13812345678，座机：010-12345678"
    phones = re.findall(phone_pattern, text)
    print(f"手机号: {phones}")

    # IP 地址
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    text = "服务器 192.168.1.1 和 10.0.0.255"
    ips = re.findall(ip_pattern, text)
    print(f"IP 地址: {ips}")


if __name__ == "__main__":
    demo_basic_matching()
    demo_findall()
    demo_substitution()
    demo_split()
    demo_compile()
    demo_groups()
    demo_greedy()
    demo_common_patterns()

    print("\n✅ re 演示完成!")


