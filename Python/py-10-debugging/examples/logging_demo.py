#!/usr/bin/env python3
"""logging 最佳实践演示

运行方式：
python logging_demo.py
LOG_LEVEL=DEBUG python logging_demo.py
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


# ============================================================
# 基本配置
# ============================================================

def demo_basic_logging():
    """基本日志配置"""
    print("\n1. 基本日志配置")
    print("-" * 30)

    # 重置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 演示各级别
    logging.debug("这是 DEBUG 级别")
    logging.info("这是 INFO 级别")
    logging.warning("这是 WARNING 级别")
    logging.error("这是 ERROR 级别")
    logging.critical("这是 CRITICAL 级别")


# ============================================================
# 模块 Logger
# ============================================================

def demo_module_logger():
    """模块级 Logger"""
    print("\n2. 模块级 Logger")
    print("-" * 30)

    # 推荐：使用 __name__
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 如果没有 handler，添加一个
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)

    logger.info("使用模块 Logger")
    logger.debug("Logger 名称是模块名")


# ============================================================
# 多 Handler 配置
# ============================================================

def demo_multiple_handlers():
    """多 Handler 配置"""
    print("\n3. 多 Handler 配置")
    print("-" * 30)

    # 创建 Logger
    logger = logging.getLogger("multi_handler")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 控制台 Handler (INFO 及以上)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)s - %(message)s"
    ))

    # 创建日志目录
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # 文件 Handler (DEBUG 及以上)
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 测试
    logger.debug("DEBUG: 只写入文件")
    logger.info("INFO: 控制台和文件都有")
    logger.warning("WARNING: 控制台和文件都有")

    print(f"日志文件: {log_dir / 'app.log'}")


# ============================================================
# 轮转日志
# ============================================================

def demo_rotating_handler():
    """轮转日志 Handler"""
    print("\n4. 轮转日志")
    print("-" * 30)

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # 按大小轮转
    size_handler = RotatingFileHandler(
        log_dir / "size_rotate.log",
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,
    )
    print(f"按大小轮转: maxBytes=1MB, backupCount=5")

    # 按时间轮转
    time_handler = TimedRotatingFileHandler(
        log_dir / "time_rotate.log",
        when="midnight",  # 每天午夜
        interval=1,
        backupCount=7,  # 保留 7 天
    )
    print(f"按时间轮转: when=midnight, backupCount=7")


# ============================================================
# 字典配置
# ============================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "myapp": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}


def demo_dict_config():
    """字典配置"""
    print("\n5. 字典配置")
    print("-" * 30)

    logging.config.dictConfig(LOGGING_CONFIG)

    logger = logging.getLogger("myapp")
    logger.info("使用字典配置")
    logger.debug("DEBUG 级别 (不会显示，因为 handler 是 INFO)")


# ============================================================
# 生产环境配置
# ============================================================

def setup_production_logging():
    """生产环境日志配置"""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # 重置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 降低第三方库日志级别
    for name in ["urllib3", "requests", "httpx", "httpcore"]:
        logging.getLogger(name).setLevel(logging.WARNING)


def demo_production():
    """生产环境配置演示"""
    print("\n6. 生产环境配置")
    print("-" * 30)
    print(f"LOG_LEVEL={os.environ.get('LOG_LEVEL', 'INFO')}")

    setup_production_logging()

    logger = logging.getLogger("production_app")
    logger.info("生产环境日志")
    logger.debug("这条在 INFO 级别下不会显示")


# ============================================================
# 日志最佳实践
# ============================================================

def demo_best_practices():
    """日志最佳实践"""
    print("\n7. 最佳实践")
    print("-" * 30)

    logger = logging.getLogger("best_practices")

    # 1. 使用参数化消息（性能更好）
    user_id = 123
    action = "login"

    # ✓ 好：延迟格式化
    logger.info("用户 %s 执行 %s", user_id, action)

    # ✗ 不好：立即格式化
    # logger.info(f"用户 {user_id} 执行 {action}")

    # 2. 记录异常信息
    try:
        x = 1 / 0
    except ZeroDivisionError:
        logger.exception("计算错误")  # 自动包含堆栈

    # 3. 使用 extra 添加上下文
    logger.info("操作完成", extra={"user_id": 123, "duration_ms": 150})


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("logging 最佳实践演示")
    print("=" * 50)

    demo_basic_logging()
    demo_module_logger()
    demo_multiple_handlers()
    demo_rotating_handler()
    demo_dict_config()
    demo_production()
    demo_best_practices()

    print("\n演示完成!")

