#!/usr/bin/env python3
"""logging 模块演示"""

import logging
import sys
from pathlib import Path


def demo_basic_logging():
    """基本日志"""
    print("=" * 50)
    print("1. 基本日志")
    print("=" * 50)

    # 创建专用 logger（避免影响其他演示）
    logger = logging.getLogger("demo_basic")
    logger.setLevel(logging.DEBUG)

    # 添加控制台 handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 输出各级别日志
    logger.debug("这是 DEBUG 信息")
    logger.info("这是 INFO 信息")
    logger.warning("这是 WARNING 信息")
    logger.error("这是 ERROR 信息")
    logger.critical("这是 CRITICAL 信息")

    # 清理
    logger.removeHandler(handler)


def demo_formatting():
    """格式化日志"""
    print("\n" + "=" * 50)
    print("2. 格式化日志")
    print("=" * 50)

    logger = logging.getLogger("demo_format")
    logger.setLevel(logging.DEBUG)

    # 详细格式
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("带详细格式的日志")
    logger.warning("警告信息")

    logger.removeHandler(handler)


def demo_exception_logging():
    """异常日志"""
    print("\n" + "=" * 50)
    print("3. 异常日志")
    print("=" * 50)

    logger = logging.getLogger("demo_exception")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        result = 1 / 0
    except Exception:
        # exception() 自动记录堆栈
        logger.exception("发生除零错误")

    print("\n使用 exc_info=True:")
    try:
        int("not a number")
    except ValueError:
        logger.error("转换错误", exc_info=True)

    logger.removeHandler(handler)


def demo_file_logging():
    """文件日志"""
    print("\n" + "=" * 50)
    print("4. 文件日志")
    print("=" * 50)

    log_file = Path("demo.log")

    logger = logging.getLogger("demo_file")
    logger.setLevel(logging.DEBUG)

    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.debug("DEBUG 信息（只写入文件）")
    logger.info("INFO 信息（写入文件和控制台）")
    logger.warning("WARNING 信息")

    # 清理
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
    file_handler.close()

    # 显示日志文件内容
    print(f"\n日志文件内容 ({log_file}):")
    print(log_file.read_text())

    # 删除日志文件
    log_file.unlink()


def demo_multiple_handlers():
    """多个 Handler"""
    print("\n" + "=" * 50)
    print("5. 多个 Handler")
    print("=" * 50)

    logger = logging.getLogger("demo_multi")
    logger.setLevel(logging.DEBUG)

    # 控制台 handler - 只显示 WARNING+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("控制台: %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # 文件 handler - 记录所有
    log_file = Path("multi_demo.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("文件: %(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.debug("DEBUG - 只写入文件")
    logger.info("INFO - 只写入文件")
    logger.warning("WARNING - 写入文件和控制台")
    logger.error("ERROR - 写入文件和控制台")

    # 清理
    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)
    file_handler.close()

    print(f"\n日志文件内容:")
    print(log_file.read_text())
    log_file.unlink()


def demo_best_practices():
    """最佳实践"""
    print("\n" + "=" * 50)
    print("6. 最佳实践")
    print("=" * 50)

    # 使用模块名作为 logger 名称
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 使用 % 格式而非 f-string（懒惰求值）
    expensive_value = "expensive_computation_result"
    logger.debug("Processing: %s", expensive_value)  # 推荐
    # logger.debug(f"Processing: {expensive_value}")  # 不推荐

    # 传递额外上下文
    logger.info("用户操作", extra={"user_id": 123})

    logger.removeHandler(handler)


def setup_logging_example():
    """配置示例"""
    print("\n" + "=" * 50)
    print("7. 配置示例")
    print("=" * 50)

    def setup_logging(level="INFO", log_file=None):
        """设置日志配置"""
        root_logger = logging.getLogger("app")
        root_logger.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        return root_logger

    # 使用
    logger = setup_logging(level="DEBUG")
    logger.debug("调试信息")
    logger.info("应用启动")
    logger.warning("警告信息")

    # 清理所有 handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


if __name__ == "__main__":
    demo_basic_logging()
    demo_formatting()
    demo_exception_logging()
    demo_file_logging()
    demo_multiple_handlers()
    demo_best_practices()
    setup_logging_example()

    print("\n✅ logging 演示完成!")


