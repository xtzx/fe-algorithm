"""核心功能模块"""


def greet(name: str, greeting: str = "Hello") -> str:
    """
    生成问候语

    Args:
        name: 要问候的名字
        greeting: 问候语前缀，默认 "Hello"

    Returns:
        完整的问候语字符串

    Examples:
        >>> greet("World")
        'Hello, World!'
        >>> greet("Python", "Hi")
        'Hi, Python!'
    """
    return f"{greeting}, {name}!"


def calculate(a: float, b: float, operation: str = "add") -> float:
    """
    执行基本数学运算

    Args:
        a: 第一个操作数
        b: 第二个操作数
        operation: 运算类型 (add, sub, mul, div)

    Returns:
        运算结果

    Raises:
        ValueError: 不支持的运算类型
        ZeroDivisionError: 除以零

    Examples:
        >>> calculate(1, 2)
        3.0
        >>> calculate(10, 3, "sub")
        7.0
    """
    operations = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
    }

    if operation not in operations:
        raise ValueError(f"不支持的运算: {operation}")

    return float(operations[operation](a, b))

