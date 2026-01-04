#!/usr/bin/env python3
"""import 系统演示"""

import sys
import importlib
import importlib.util


def demo_sys_path():
    """sys.path 演示"""
    print("=" * 50)
    print("1. sys.path 模块查找路径")
    print("=" * 50)

    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")


def demo_module_info():
    """模块信息演示"""
    print("\n" + "=" * 50)
    print("2. 模块信息")
    print("=" * 50)

    import json

    print(f"模块名: {json.__name__}")
    print(f"文件位置: {json.__file__}")
    print(f"包: {json.__package__}")
    print(f"文档: {json.__doc__[:100]}...")

    # 缓存位置
    if hasattr(json, '__cached__'):
        print(f"缓存: {json.__cached__}")


def demo_dynamic_import():
    """动态导入演示"""
    print("\n" + "=" * 50)
    print("3. 动态导入")
    print("=" * 50)

    # 使用 importlib.import_module
    module_name = "json"
    json_module = importlib.import_module(module_name)

    print(f"动态导入 {module_name}")
    result = json_module.dumps({"a": 1, "b": 2})
    print(f"json.dumps 结果: {result}")

    # 导入子模块
    os_path = importlib.import_module("os.path")
    print(f"os.path.exists('.'): {os_path.exists('.')}")


def demo_reload():
    """模块重载演示"""
    print("\n" + "=" * 50)
    print("4. 模块重载")
    print("=" * 50)

    import json

    # 获取原始 dumps
    original_dumps = json.dumps

    # 重载模块
    importlib.reload(json)

    # 检查是否是同一个函数对象
    print(f"重载后 dumps 是同一对象: {json.dumps is original_dumps}")
    print("(重载会创建新的模块对象)")


def demo_check_module():
    """检查模块是否存在"""
    print("\n" + "=" * 50)
    print("5. 检查模块是否存在")
    print("=" * 50)

    def module_exists(name):
        spec = importlib.util.find_spec(name)
        return spec is not None

    modules = ["json", "numpy", "requests", "nonexistent_module"]

    for mod in modules:
        exists = module_exists(mod)
        print(f"{mod}: {'存在' if exists else '不存在'}")


def demo_name_main():
    """__name__ 演示"""
    print("\n" + "=" * 50)
    print("6. __name__ 变量")
    print("=" * 50)

    print(f"当前模块 __name__: {__name__}")

    # 检查导入的模块
    import json
    print(f"json.__name__: {json.__name__}")

    print("\n说明:")
    print("- 直接运行时 __name__ = '__main__'")
    print("- 被导入时 __name__ = 模块名")


def demo_all_exports():
    """__all__ 演示"""
    print("\n" + "=" * 50)
    print("7. __all__ 控制导出")
    print("=" * 50)

    import os

    if hasattr(os, '__all__'):
        print(f"os.__all__ (前10个): {os.__all__[:10]}")
    else:
        print("os 没有定义 __all__")

    print("\n说明:")
    print("__all__ 定义 from module import * 时导出的名称")


def demo_lazy_import():
    """延迟导入演示"""
    print("\n" + "=" * 50)
    print("8. 延迟导入模式")
    print("=" * 50)

    def heavy_function():
        # 只在需要时导入
        import json
        return json.dumps({"data": "value"})

    print("延迟导入: 只在函数调用时才导入模块")
    result = heavy_function()
    print(f"结果: {result}")


if __name__ == "__main__":
    demo_sys_path()
    demo_module_info()
    demo_dynamic_import()
    demo_reload()
    demo_check_module()
    demo_name_main()
    demo_all_exports()
    demo_lazy_import()

    print("\n✅ import 系统演示完成!")

