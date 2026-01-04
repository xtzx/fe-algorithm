# sample_bad.py - 故意包含多个代码质量问题
# 这个文件用于演示 ruff 和 pyright 能检测到的问题
# 注意：这个文件在 pyproject.toml 中被排除检查

# 问题 1: 未使用的 import (F401)
import os
import sys
import json
from pathlib import Path

# 问题 2: import 未排序 (I001)
import requests
import collections
import abc

# 问题 3: 未使用的变量 (F841)
unused_variable = 42

# 问题 4: 缺少类型注解
def add(a, b):
    return a + b

# 问题 5: 变量命名不规范 (N806 - 函数中使用大写变量)
def bad_naming():
    MyVariable = 1
    return MyVariable

# 问题 6: 行太长 (E501)
very_long_line = "This is a very long line that exceeds the maximum line length limit of 88 characters and should be broken up into multiple lines for better readability"

# 问题 7: 缺少文档字符串 (D100, D103)
class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

# 问题 8: 比较应该用 is None (E711)
def check_none(x):
    if x == None:
        return True
    return False

# 问题 9: 不必要的 pass (PIE790)
def empty_function():
    pass
    pass

# 问题 10: 可变默认参数 (B006)
def bad_default(items=[]):
    items.append(1)
    return items

# 问题 11: 裸 except (E722)
def catch_all():
    try:
        risky_operation()
    except:
        pass

def risky_operation():
    pass

# 问题 12: f-string 没有占位符 (F541)
message = f"Hello World"

# 问题 13: 使用 == 比较 True/False (E712)
def check_bool(x):
    if x == True:
        return "yes"
    return "no"

# 问题 14: 类型注解错误 (pyright)
def typed_function(x: int) -> str:
    return x  # 返回类型不匹配

# 问题 15: 可能的 None 访问 (pyright)
def maybe_none(x: str | None) -> int:
    return len(x)  # x 可能是 None

# 问题 16: 不必要的列表推导式 (C400)
result = list([x for x in range(10)])

# 问题 17: 可以用字典推导式 (C402)
dict_result = dict([(x, x*2) for x in range(10)])

# 问题 18: 重复的键 (F601)
duplicate_dict = {
    "key": 1,
    "key": 2,
}

# 问题 19: 星号导入 (F403)
# from some_module import *

# 问题 20: 格式化问题（由 black/ruff format 修复）
x=1
y  =  2
z=   3

def poorly_formatted(a,b,c):
    return a+b+c

