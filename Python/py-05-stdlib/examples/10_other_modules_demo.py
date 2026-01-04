#!/usr/bin/env python3
"""其他常用模块演示"""

import random
import secrets
import uuid
import hashlib
import base64
from urllib.parse import urlparse, urlencode, parse_qs, quote, unquote


def demo_random():
    """random 模块"""
    print("=" * 50)
    print("1. random 模块")
    print("=" * 50)

    # 设置种子（可复现）
    random.seed(42)

    # 随机浮点数 [0, 1)
    print(f"random(): {random.random()}")

    # 指定范围
    print(f"uniform(1.0, 10.0): {random.uniform(1.0, 10.0)}")
    print(f"randint(1, 100): {random.randint(1, 100)}")
    print(f"randrange(0, 100, 10): {random.randrange(0, 100, 10)}")

    # 序列操作
    items = [1, 2, 3, 4, 5]
    print(f"choice({items}): {random.choice(items)}")
    print(f"choices({items}, k=3): {random.choices(items, k=3)}")
    print(f"sample({items}, k=3): {random.sample(items, k=3)}")

    # 打乱顺序
    shuffled = items.copy()
    random.shuffle(shuffled)
    print(f"shuffle({items}): {shuffled}")


def demo_secrets():
    """secrets 模块 - 安全随机"""
    print("\n" + "=" * 50)
    print("2. secrets 模块 - 安全随机")
    print("=" * 50)

    # 随机字节
    token_bytes = secrets.token_bytes(16)
    print(f"token_bytes(16): {token_bytes[:10]}... (长度: {len(token_bytes)})")

    # 十六进制字符串
    token_hex = secrets.token_hex(16)
    print(f"token_hex(16): {token_hex}")

    # URL 安全字符串
    token_url = secrets.token_urlsafe(16)
    print(f"token_urlsafe(16): {token_url}")

    # 随机整数
    n = secrets.randbelow(100)
    print(f"randbelow(100): {n}")

    # 安全选择
    items = ["apple", "banana", "cherry"]
    choice = secrets.choice(items)
    print(f"choice({items}): {choice}")

    # 比较字符串（防止时序攻击）
    a = "password123"
    b = "password123"
    print(f"compare_digest: {secrets.compare_digest(a, b)}")


def demo_uuid():
    """uuid 模块"""
    print("\n" + "=" * 50)
    print("3. uuid 模块")
    print("=" * 50)

    # UUID4（随机生成，最常用）
    id1 = uuid.uuid4()
    print(f"uuid4(): {id1}")

    # UUID1（基于时间和 MAC 地址）
    id2 = uuid.uuid1()
    print(f"uuid1(): {id2}")

    # 属性
    print(f"  hex: {id1.hex}")
    print(f"  bytes 长度: {len(id1.bytes)}")
    print(f"  version: {id1.version}")

    # 从字符串创建
    id3 = uuid.UUID(str(id1))
    print(f"从字符串创建: {id3}")
    print(f"相等性: {id1 == id3}")


def demo_hashlib():
    """hashlib 模块"""
    print("\n" + "=" * 50)
    print("4. hashlib 模块")
    print("=" * 50)

    data = b"Hello, World!"

    # MD5（不安全，仅用于校验）
    md5 = hashlib.md5(data)
    print(f"MD5: {md5.hexdigest()}")

    # SHA256（推荐）
    sha256 = hashlib.sha256(data)
    print(f"SHA256: {sha256.hexdigest()}")

    # SHA512
    sha512 = hashlib.sha512(data)
    print(f"SHA512: {sha512.hexdigest()[:32]}... (长度: {len(sha512.hexdigest())})")

    # 增量更新
    h = hashlib.sha256()
    h.update(b"Hello, ")
    h.update(b"World!")
    print(f"增量更新: {h.hexdigest()}")

    # 密码哈希（使用 pbkdf2）
    password = b"mysecretpassword"
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", password, salt, 100000)
    print(f"PBKDF2: {key.hex()[:32]}...")


def demo_base64():
    """base64 模块"""
    print("\n" + "=" * 50)
    print("5. base64 模块")
    print("=" * 50)

    data = b"Hello, World!"

    # 编码
    encoded = base64.b64encode(data)
    print(f"原始数据: {data}")
    print(f"b64encode: {encoded}")

    # 解码
    decoded = base64.b64decode(encoded)
    print(f"b64decode: {decoded}")

    # URL 安全编码
    url_safe = base64.urlsafe_b64encode(data)
    print(f"urlsafe_b64encode: {url_safe}")

    # 字符串编码
    text = "你好，世界！"
    encoded_str = base64.b64encode(text.encode("utf-8")).decode("ascii")
    print(f"字符串编码: {encoded_str}")

    # 解码回来
    decoded_str = base64.b64decode(encoded_str).decode("utf-8")
    print(f"字符串解码: {decoded_str}")


def demo_urllib():
    """urllib 模块"""
    print("\n" + "=" * 50)
    print("6. urllib.parse 模块")
    print("=" * 50)

    # 解析 URL
    url = "https://example.com:8080/path/to/page?name=alice&age=25#section"
    result = urlparse(url)

    print(f"URL: {url}")
    print(f"  scheme: {result.scheme}")
    print(f"  netloc: {result.netloc}")
    print(f"  path: {result.path}")
    print(f"  query: {result.query}")
    print(f"  fragment: {result.fragment}")

    # 解析查询参数
    params = parse_qs(result.query)
    print(f"  params: {params}")

    # 构建查询字符串
    query = urlencode({"name": "张三", "age": 25, "city": "北京"})
    print(f"\nurlencode: {query}")

    # URL 编解码
    text = "你好 世界"
    encoded = quote(text)
    print(f"quote('{text}'): {encoded}")

    decoded = unquote(encoded)
    print(f"unquote: {decoded}")


def demo_practical():
    """实际应用"""
    print("\n" + "=" * 50)
    print("7. 实际应用")
    print("=" * 50)

    # 生成 API Token
    def generate_api_token(user_id: str) -> str:
        random_part = secrets.token_urlsafe(24)
        return f"{user_id}_{random_part}"

    token = generate_api_token("user123")
    print(f"API Token: {token}")

    # 文件哈希校验
    def calculate_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    file_content = b"This is file content"
    file_hash = calculate_hash(file_content)
    print(f"文件哈希: {file_hash}")

    # 生成会话 ID
    session_id = str(uuid.uuid4())
    print(f"会话 ID: {session_id}")

    # 密码哈希存储
    def hash_password(password: str) -> tuple[str, str]:
        salt = secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000
        )
        return salt, key.hex()

    salt, hashed = hash_password("mypassword123")
    print(f"密码哈希: salt={salt[:16]}..., hash={hashed[:16]}...")


if __name__ == "__main__":
    demo_random()
    demo_secrets()
    demo_uuid()
    demo_hashlib()
    demo_base64()
    demo_urllib()
    demo_practical()

    print("\n✅ 其他模块演示完成!")


