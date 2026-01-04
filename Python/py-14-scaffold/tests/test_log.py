"""
测试日志模块
"""

import logging
import pytest

from scaffold.log import (
    setup_logging,
    get_logger,
    get_context_logger,
    TextFormatter,
    JsonFormatter,
)


class TestSetupLogging:
    """测试日志配置"""

    def test_setup_default(self):
        """测试默认配置"""
        setup_logging()

        logger = logging.getLogger()
        assert logger.level == logging.INFO

    def test_setup_debug(self):
        """测试 DEBUG 级别"""
        setup_logging(level="DEBUG")

        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

    def test_setup_json_format(self):
        """测试 JSON 格式"""
        setup_logging(format="json")

        logger = logging.getLogger()
        handler = logger.handlers[-1]
        assert isinstance(handler.formatter, JsonFormatter)


class TestGetLogger:
    """测试获取 logger"""

    def test_get_logger(self):
        """测试获取 logger"""
        logger = get_logger(__name__)

        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__

    def test_get_logger_none(self):
        """测试获取根 logger"""
        logger = get_logger(None)

        assert isinstance(logger, logging.Logger)


class TestContextLogger:
    """测试上下文 logger"""

    def test_context_logger(self, caplog):
        """测试带上下文的日志"""
        setup_logging(level="DEBUG")

        logger = get_context_logger(
            __name__,
            request_id="abc123",
            user_id="user1",
        )

        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        assert "request_id=abc123" in caplog.text
        assert "user_id=user1" in caplog.text
        assert "Test message" in caplog.text


class TestTextFormatter:
    """测试文本格式化器"""

    def test_format_info(self):
        """测试 INFO 级别格式化"""
        formatter = TextFormatter(use_colors=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "Test message" in output

    def test_format_with_exception(self):
        """测试异常格式化"""
        formatter = TextFormatter(use_colors=False)

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)

        assert "ERROR" in output
        assert "ValueError" in output
        assert "Test error" in output


class TestJsonFormatter:
    """测试 JSON 格式化器"""

    def test_format_json(self):
        """测试 JSON 格式化"""
        import json

        formatter = JsonFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data

