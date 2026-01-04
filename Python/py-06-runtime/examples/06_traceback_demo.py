#!/usr/bin/env python3
"""错误与 Traceback 演示"""

import traceback
import sys


def demo_basic_exception():
    """基本异常演示"""
    print("=" * 50)
    print("1. 基本异常处理")
    print("=" * 50)

    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        print(f"捕获异常: {type(e).__name__}: {e}")


def demo_multiple_exceptions():
    """多异常处理演示"""
    print("\n" + "=" * 50)
    print("2. 多异常处理")
    print("=" * 50)

    def risky_operation(value):
        try:
            if value == "type":
                return "10" + 10
            elif value == "key":
                return {}["missing"]
            elif value == "index":
                return [1, 2, 3][10]
            else:
                return int(value)
        except (TypeError, ValueError) as e:
            print(f"类型或值错误: {e}")
        except KeyError as e:
            print(f"键不存在: {e}")
        except IndexError as e:
            print(f"索引越界: {e}")

    risky_operation("type")
    risky_operation("key")
    risky_operation("index")


def demo_exception_chain():
    """异常链演示"""
    print("\n" + "=" * 50)
    print("3. 异常链 (raise from)")
    print("=" * 50)

    def parse_number(s):
        try:
            return int(s)
        except ValueError as e:
            raise RuntimeError(f"无法解析数字: {s}") from e

    try:
        parse_number("abc")
    except RuntimeError as e:
        print(f"异常: {e}")
        print(f"原因: {e.__cause__}")


def demo_traceback_info():
    """Traceback 信息演示"""
    print("\n" + "=" * 50)
    print("4. Traceback 信息")
    print("=" * 50)

    def level3():
        return 1 / 0

    def level2():
        return level3()

    def level1():
        return level2()

    try:
        level1()
    except Exception as e:
        print("--- 简单信息 ---")
        print(f"类型: {type(e).__name__}")
        print(f"消息: {e}")

        print("\n--- 完整 Traceback ---")
        print(traceback.format_exc())


def demo_custom_exception():
    """自定义异常演示"""
    print("\n" + "=" * 50)
    print("5. 自定义异常")
    print("=" * 50)

    class ValidationError(Exception):
        def __init__(self, field, message):
            self.field = field
            self.message = message
            super().__init__(f"{field}: {message}")

    class AgeValidationError(ValidationError):
        pass

    def validate_age(age):
        if not isinstance(age, int):
            raise ValidationError("age", "必须是整数")
        if age < 0 or age > 150:
            raise AgeValidationError("age", f"必须在 0-150 之间，当前值: {age}")
        return True

    try:
        validate_age("twenty")
    except ValidationError as e:
        print(f"验证失败 - 字段: {e.field}, 原因: {e.message}")

    try:
        validate_age(200)
    except AgeValidationError as e:
        print(f"年龄验证失败 - {e.message}")


def demo_exception_hierarchy():
    """异常层级演示"""
    print("\n" + "=" * 50)
    print("6. 异常层级")
    print("=" * 50)

    exceptions = [
        ZeroDivisionError,
        TypeError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PermissionError,
    ]

    for exc in exceptions:
        mro = [c.__name__ for c in exc.__mro__ if issubclass(c, BaseException)]
        print(f"{exc.__name__}: {' -> '.join(mro)}")


def demo_try_else_finally():
    """try-else-finally 演示"""
    print("\n" + "=" * 50)
    print("7. try-else-finally")
    print("=" * 50)

    def divide(a, b):
        try:
            result = a / b
        except ZeroDivisionError:
            print("  except: 除零错误")
            return None
        else:
            print(f"  else: 计算成功，结果 = {result}")
            return result
        finally:
            print("  finally: 清理资源（总是执行）")

    print("正常情况:")
    divide(10, 2)

    print("\n异常情况:")
    divide(10, 0)


def demo_context_manager():
    """上下文管理器异常处理演示"""
    print("\n" + "=" * 50)
    print("8. 上下文管理器异常处理")
    print("=" * 50)

    class ManagedResource:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            print(f"  进入: {self.name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print(f"  退出: {self.name}")
            if exc_type:
                print(f"  异常: {exc_type.__name__}: {exc_val}")
                return True  # 抑制异常
            return False

    print("正常使用:")
    with ManagedResource("资源A") as r:
        print(f"  使用 {r.name}")

    print("\n异常时（被抑制）:")
    with ManagedResource("资源B") as r:
        raise ValueError("测试异常")
    print("  程序继续执行")


def demo_suppress():
    """suppress 上下文管理器演示"""
    print("\n" + "=" * 50)
    print("9. contextlib.suppress")
    print("=" * 50)

    from contextlib import suppress

    # 常规写法
    try:
        result = int("abc")
    except ValueError:
        pass

    # 使用 suppress
    with suppress(ValueError, TypeError):
        result = int("abc")

    print("使用 suppress 简化异常忽略")


if __name__ == "__main__":
    demo_basic_exception()
    demo_multiple_exceptions()
    demo_exception_chain()
    demo_traceback_info()
    demo_custom_exception()
    demo_exception_hierarchy()
    demo_try_else_finally()
    demo_context_manager()
    demo_suppress()

    print("\n✅ 错误与 Traceback 演示完成!")

