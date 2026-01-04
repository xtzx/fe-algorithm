#!/usr/bin/env python3
"""
脚手架使用示例

演示如何使用配置、日志和工具函数
"""

import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scaffold import __version__, get_settings, setup_logging, get_logger
from scaffold.utils import timer, retry, format_duration


def example_config():
    """示例：配置管理"""
    print("=" * 60)
    print("示例 1: 配置管理")
    print("=" * 60)

    settings = get_settings()

    print(f"应用名称: {settings.app_name}")
    print(f"应用环境: {settings.app_env}")
    print(f"调试模式: {settings.debug}")
    print(f"日志级别: {settings.log_level}")
    print(f"是否生产: {settings.is_production}")
    print()

    # JSON 格式
    print("完整配置 (JSON):")
    print(settings.model_dump_json(indent=2))


def example_logging():
    """示例：日志配置"""
    print("\n" + "=" * 60)
    print("示例 2: 日志配置")
    print("=" * 60)

    # 配置日志
    setup_logging(level="DEBUG", format="text")

    # 获取 logger
    logger = get_logger(__name__)

    # 各级别日志
    logger.debug("这是 DEBUG 级别日志")
    logger.info("这是 INFO 级别日志")
    logger.warning("这是 WARNING 级别日志")
    logger.error("这是 ERROR 级别日志")


def example_decorators():
    """示例：装饰器"""
    print("\n" + "=" * 60)
    print("示例 3: 装饰器")
    print("=" * 60)

    # 计时装饰器
    @timer
    def slow_function():
        import time

        time.sleep(0.1)
        return "done"

    print("计时装饰器:")
    result = slow_function()
    print(f"结果: {result}")
    print()

    # 重试装饰器
    attempt_count = 0

    @retry(max_attempts=3, delay=0.1)
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Attempt {attempt_count} failed")
        return "success"

    print("重试装饰器:")
    try:
        result = flaky_function()
        print(f"结果: {result} (尝试次数: {attempt_count})")
    except Exception as e:
        print(f"失败: {e}")


def example_utils():
    """示例：工具函数"""
    print("\n" + "=" * 60)
    print("示例 4: 工具函数")
    print("=" * 60)

    from scaffold.utils import (
        snake_to_camel,
        camel_to_snake,
        truncate,
        chunk_list,
        unique,
    )

    # 字符串转换
    print("字符串转换:")
    print(f"  snake_to_camel('hello_world') = '{snake_to_camel('hello_world')}'")
    print(f"  camel_to_snake('helloWorld') = '{camel_to_snake('helloWorld')}'")
    print()

    # 字符串截断
    print("字符串截断:")
    long_text = "This is a very long text that needs to be truncated"
    print(f"  原文: '{long_text}'")
    print(f"  截断: '{truncate(long_text, 30)}'")
    print()

    # 持续时间格式化
    print("持续时间格式化:")
    for seconds in [0.5, 5.5, 65, 3700]:
        print(f"  {seconds}s = '{format_duration(seconds)}'")
    print()

    # 列表操作
    print("列表操作:")
    print(f"  chunk_list([1,2,3,4,5], 2) = {chunk_list([1, 2, 3, 4, 5], 2)}")
    print(f"  unique([1,2,2,3,1,4]) = {unique([1, 2, 2, 3, 1, 4])}")


def main():
    """运行所有示例"""
    print(f"Scaffold 脚手架示例 v{__version__}")
    print()

    example_config()
    example_logging()
    example_decorators()
    example_utils()

    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

