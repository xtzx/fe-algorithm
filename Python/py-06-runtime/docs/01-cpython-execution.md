# 01. CPython 执行链路

## 本节目标

- 理解 Python 代码从源码到执行的全过程
- 了解 AST 和字节码
- 对比 JavaScript 执行模型

---

## 执行流程概览

```
源代码 (.py)
    ↓ 词法分析 (Tokenizer)
Token 流
    ↓ 语法分析 (Parser)
AST (抽象语法树)
    ↓ 编译 (Compiler)
字节码 (Code Object)
    ↓ 解释执行
Python 虚拟机 (PVM)
```

---

## 1. 词法分析与语法分析

### 查看 Token

```python
import tokenize
import io

code = "x = 1 + 2"
tokens = tokenize.generate_tokens(io.StringIO(code).readline)

for tok in tokens:
    print(tok)

# TokenInfo(type=1 (NAME), string='x', ...)
# TokenInfo(type=54 (OP), string='=', ...)
# TokenInfo(type=2 (NUMBER), string='1', ...)
# ...
```

### 查看 AST

```python
import ast

code = """
def greet(name):
    return f"Hello, {name}"
"""

tree = ast.parse(code)
print(ast.dump(tree, indent=2))

# Module(
#   body=[
#     FunctionDef(
#       name='greet',
#       args=arguments(...),
#       body=[Return(value=JoinedStr(...))]
#     )
#   ]
# )
```

### 遍历和修改 AST

```python
import ast

class NameVisitor(ast.NodeVisitor):
    def visit_Name(self, node):
        print(f"变量名: {node.id}")
        self.generic_visit(node)

code = "x = y + z"
tree = ast.parse(code)
NameVisitor().visit(tree)
# 变量名: x
# 变量名: y
# 变量名: z
```

---

## 2. 编译为字节码

### compile() 函数

```python
code = "x = 1 + 2"

# 编译为代码对象
code_obj = compile(code, "<string>", "exec")

print(type(code_obj))  # <class 'code'>
print(code_obj.co_code)  # 字节码（二进制）
print(code_obj.co_consts)  # 常量表
print(code_obj.co_names)  # 名称表
```

### 查看字节码

```python
import dis

code = """
def add(a, b):
    return a + b
"""

exec(code)
dis.dis(add)

#   2           0 LOAD_FAST                0 (a)
#               2 LOAD_FAST                1 (b)
#               4 BINARY_ADD
#               6 RETURN_VALUE
```

### 字节码指令解释

| 指令 | 说明 |
|------|------|
| `LOAD_FAST` | 加载局部变量到栈 |
| `LOAD_CONST` | 加载常量到栈 |
| `BINARY_ADD` | 栈顶两元素相加 |
| `STORE_FAST` | 存储到局部变量 |
| `RETURN_VALUE` | 返回栈顶值 |
| `CALL_FUNCTION` | 调用函数 |

---

## 3. Python 虚拟机

Python 虚拟机是**基于栈**的虚拟机：

```python
# x = 1 + 2 的执行过程

# 字节码：
# LOAD_CONST 1    # 栈: [1]
# LOAD_CONST 2    # 栈: [1, 2]
# BINARY_ADD      # 栈: [3]
# STORE_NAME x    # 栈: [], x=3
```

### 查看执行过程

```python
import dis

def example():
    x = 1
    y = 2
    z = x + y
    return z

dis.dis(example)
```

输出：
```
  2           0 LOAD_CONST               1 (1)
              2 STORE_FAST               0 (x)

  3           4 LOAD_CONST               2 (2)
              6 STORE_FAST               1 (y)

  4           8 LOAD_FAST                0 (x)
             10 LOAD_FAST                1 (y)
             12 BINARY_ADD
             14 STORE_FAST               2 (z)

  5          16 LOAD_FAST                2 (z)
             18 RETURN_VALUE
```

---

## 4. 与 JavaScript V8 对比

| 特性 | CPython | V8 (JavaScript) |
|------|---------|-----------------|
| **执行方式** | 解释执行字节码 | JIT 编译为机器码 |
| **预编译** | .pyc 缓存 | 无（运行时编译） |
| **优化** | 几乎不优化 | 多级优化（Ignition → TurboFan） |
| **速度** | 较慢 | 较快 |
| **启动** | 快 | 首次较慢（JIT 预热） |

### 为什么 Python 不用 JIT？

1. **动态性太强**：类型随时可变
2. **历史原因**：CPython 设计时代久远
3. **替代方案**：PyPy、Numba 提供 JIT

---

## 5. PyPy 与其他实现

### CPython vs PyPy

```
CPython:  源码 → 字节码 → 解释执行
PyPy:     源码 → 字节码 → JIT编译 → 机器码
```

| 特性 | CPython | PyPy |
|------|---------|------|
| 启动速度 | 快 | 慢 |
| 运行速度 | 慢 | 快 (3-10x) |
| 内存 | 较少 | 较多 |
| C 扩展 | 完美支持 | 有限支持 |

### 其他 Python 实现

- **PyPy**: JIT 编译，性能好
- **Jython**: 运行在 JVM
- **IronPython**: 运行在 .NET
- **MicroPython**: 嵌入式设备
- **GraalPy**: GraalVM 上的 Python

---

## 6. 实际应用

### 代码检查工具

```python
import ast

def check_no_print(code):
    """检查代码中是否使用了 print"""
    tree = ast.parse(code)

    class PrintChecker(ast.NodeVisitor):
        def __init__(self):
            self.has_print = False

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'print':
                self.has_print = True
            self.generic_visit(node)

    checker = PrintChecker()
    checker.visit(tree)
    return not checker.has_print

code = """
def hello():
    print("hi")
"""
print(check_no_print(code))  # False
```

### 字节码优化分析

```python
import dis

# 比较两种写法的字节码
def v1():
    return 1 + 2 + 3

def v2():
    return 6

print("v1 字节码:")
dis.dis(v1)
print("\nv2 字节码:")
dis.dis(v2)

# v1 会在编译时优化成常量 6
```

---

## 本节要点

1. **执行链路**: 源码 → AST → 字节码 → PVM 执行
2. **ast 模块**: 解析和分析代码结构
3. **dis 模块**: 反汇编查看字节码
4. **compile()**: 编译代码为代码对象
5. **CPython vs V8**: 解释执行 vs JIT 编译
6. **PyPy**: Python 的 JIT 实现

