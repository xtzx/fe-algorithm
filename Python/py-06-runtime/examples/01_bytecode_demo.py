#!/usr/bin/env python3
"""字节码与 AST 演示"""

import ast
import dis


def demo_ast():
    """AST 演示"""
    print("=" * 50)
    print("1. AST 解析")
    print("=" * 50)

    code = """
def greet(name):
    return f"Hello, {name}"
"""

    tree = ast.parse(code)
    print("AST 结构:")
    print(ast.dump(tree, indent=2))

    # 遍历 AST
    print("\n遍历所有函数定义:")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            print(f"  函数名: {node.name}")


def demo_dis():
    """字节码反汇编演示"""
    print("\n" + "=" * 50)
    print("2. 字节码反汇编")
    print("=" * 50)

    def add(a, b):
        return a + b

    print("函数 add(a, b) 的字节码:")
    dis.dis(add)

    # 更复杂的例子
    print("\n条件语句的字节码:")

    def compare(x, y):
        if x > y:
            return x
        else:
            return y

    dis.dis(compare)


def demo_code_object():
    """Code Object 演示"""
    print("\n" + "=" * 50)
    print("3. Code Object 属性")
    print("=" * 50)

    def example(a, b, c=10):
        """示例函数"""
        x = a + b + c
        return x * 2

    code = example.__code__

    print(f"co_name (函数名): {code.co_name}")
    print(f"co_argcount (参数数量): {code.co_argcount}")
    print(f"co_varnames (变量名): {code.co_varnames}")
    print(f"co_consts (常量): {code.co_consts}")
    print(f"co_stacksize (栈大小): {code.co_stacksize}")


def demo_compile():
    """compile() 函数演示"""
    print("\n" + "=" * 50)
    print("4. compile() 动态编译")
    print("=" * 50)

    source = """
result = 0
for i in range(5):
    result += i
print(f"结果: {result}")
"""

    # 编译为代码对象
    code_obj = compile(source, "<string>", "exec")

    print("编译成功，执行代码:")
    exec(code_obj)

    # 查看字节码
    print("\n字节码:")
    dis.dis(code_obj)


def demo_optimization():
    """编译器优化演示"""
    print("\n" + "=" * 50)
    print("5. 编译器优化")
    print("=" * 50)

    # 常量折叠
    def const_folding():
        return 1 + 2 + 3

    print("常量折叠 (1 + 2 + 3):")
    dis.dis(const_folding)
    print("注意: 编译器直接计算出 6")


if __name__ == "__main__":
    demo_ast()
    demo_dis()
    demo_code_object()
    demo_compile()
    demo_optimization()

    print("\n✅ 字节码演示完成!")

