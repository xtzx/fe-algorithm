#!/usr/bin/env python3
"""json 模块演示"""

import json
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path


def demo_basic_operations():
    """基本操作"""
    print("=" * 50)
    print("1. 基本操作")
    print("=" * 50)

    # dumps - 对象转 JSON 字符串
    data = {
        "name": "Alice",
        "age": 25,
        "skills": ["Python", "JavaScript"]
    }
    json_str = json.dumps(data)
    print(f"dumps: {json_str}")

    # loads - JSON 字符串转对象
    json_str = '{"name": "Bob", "age": 30}'
    data = json.loads(json_str)
    print(f"loads: {data}")
    print(f"访问: name={data['name']}, age={data['age']}")


def demo_file_operations():
    """文件操作"""
    print("\n" + "=" * 50)
    print("2. 文件操作")
    print("=" * 50)

    data = {"name": "Charlie", "age": 35, "active": True}

    # dump - 写入文件
    temp_file = Path("temp_demo.json")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"写入文件: {temp_file}")

    # load - 从文件读取
    with open(temp_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"读取文件: {loaded}")

    # 清理
    temp_file.unlink()


def demo_formatting():
    """格式化输出"""
    print("\n" + "=" * 50)
    print("3. 格式化输出")
    print("=" * 50)

    data = {"name": "张三", "items": [1, 2, 3], "nested": {"a": 1, "b": 2}}

    # 紧凑格式
    print("紧凑格式:")
    print(json.dumps(data))

    # 缩进格式
    print("\n缩进格式 (indent=2):")
    print(json.dumps(data, indent=2))

    # 中文不转义
    print("\n中文不转义 (ensure_ascii=False):")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # 键排序
    data = {"c": 3, "a": 1, "b": 2}
    print(f"\n键排序 (sort_keys=True):")
    print(json.dumps(data, sort_keys=True))


def demo_custom_encoder():
    """自定义编码"""
    print("\n" + "=" * 50)
    print("4. 自定义编码")
    print("=" * 50)

    # 使用 default 参数
    def custom_encoder(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    data = {
        "name": "Alice",
        "created": datetime.now()
    }

    result = json.dumps(data, default=custom_encoder, indent=2)
    print("使用 default 参数:")
    print(result)

    # 使用自定义 JSONEncoder
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return {"__datetime__": obj.isoformat()}
            return super().default(obj)

    result = json.dumps(data, cls=CustomEncoder, indent=2)
    print("\n使用自定义 JSONEncoder:")
    print(result)


def demo_custom_decoder():
    """自定义解码"""
    print("\n" + "=" * 50)
    print("5. 自定义解码")
    print("=" * 50)

    def custom_decoder(dct):
        if "__datetime__" in dct:
            return datetime.fromisoformat(dct["__datetime__"])
        return dct

    json_str = '{"name": "Alice", "created": {"__datetime__": "2024-01-15T14:30:45"}}'
    data = json.loads(json_str, object_hook=custom_decoder)

    print(f"解码结果: {data}")
    print(f"created 类型: {type(data['created'])}")


def demo_dataclass():
    """dataclass 序列化"""
    print("\n" + "=" * 50)
    print("6. dataclass 序列化")
    print("=" * 50)

    @dataclass
    class User:
        name: str
        age: int
        email: str

    user = User("Alice", 25, "alice@example.com")

    # 使用 asdict
    json_str = json.dumps(asdict(user), indent=2)
    print(f"dataclass 序列化:")
    print(json_str)


def demo_type_mapping():
    """类型对照"""
    print("\n" + "=" * 50)
    print("7. 类型对照")
    print("=" * 50)

    data = {
        "string": "hello",
        "integer": 42,
        "float": 3.14,
        "boolean_true": True,
        "boolean_false": False,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"a": 1}
    }

    json_str = json.dumps(data, indent=2)
    print("Python -> JSON:")
    print(json_str)

    # 解析回来
    parsed = json.loads(json_str)
    print("\nJSON -> Python 类型:")
    for key, value in parsed.items():
        print(f"  {key}: {type(value).__name__} = {value}")


def demo_practical_example():
    """实际应用"""
    print("\n" + "=" * 50)
    print("8. 实际应用 - 配置文件")
    print("=" * 50)

    # 默认配置
    default_config = {
        "debug": False,
        "log_level": "INFO",
        "database": {
            "host": "localhost",
            "port": 5432
        }
    }

    # 保存配置
    config_file = Path("config_demo.json")
    config_file.write_text(
        json.dumps(default_config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"保存配置到: {config_file}")

    # 读取配置
    loaded_config = json.loads(config_file.read_text(encoding="utf-8"))
    print(f"读取配置: {loaded_config}")

    # 修改配置
    loaded_config["debug"] = True
    loaded_config["database"]["host"] = "192.168.1.100"

    # 保存修改
    config_file.write_text(
        json.dumps(loaded_config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"修改后配置:")
    print(config_file.read_text())

    # 清理
    config_file.unlink()


if __name__ == "__main__":
    demo_basic_operations()
    demo_file_operations()
    demo_formatting()
    demo_custom_encoder()
    demo_custom_decoder()
    demo_dataclass()
    demo_type_mapping()
    demo_practical_example()

    print("\n✅ json 演示完成!")


