# 02. 变量与类型

## 🎯 本节目标

- 理解 Python 变量声明
- 掌握基本数据类型
- 学会类型转换和检查
- 对比 JS 的差异

---

## 📝 变量声明

### Python 方式

```python
# 直接赋值，无需关键字
name = "Alice"
age = 25
is_student = True

# 多重赋值
x, y, z = 1, 2, 3

# 交换变量（Python 特色）
a, b = b, a
```

### JS 对照

```javascript
// JS 需要 let/const/var
let name = "Alice";
const age = 25;
let isStudent = true;

// 解构赋值
let [x, y, z] = [1, 2, 3];

// 交换需要临时变量或解构
[a, b] = [b, a];
```

### ⚠️ 关键差异

| 特性 | Python | JavaScript |
|------|--------|------------|
| 声明关键字 | 无 | `let` / `const` / `var` |
| 常量 | 约定 `UPPER_CASE` | `const` |
| 变量命名 | `snake_case` | `camelCase` |
| 未声明变量 | `NameError` | `undefined`（var）或 `ReferenceError` |

---

## 🔢 基本数据类型

### 整数 (int)

```python
# Python 整数无大小限制
x = 42
big = 10 ** 100  # 非常大的数

# 不同进制
binary = 0b1010     # 二进制: 10
octal = 0o17        # 八进制: 15
hexadecimal = 0xFF  # 十六进制: 255

# 可读性分隔符
million = 1_000_000  # Python 3.6+
```

**JS 对照**：JS 的 `Number` 有精度限制，大整数用 `BigInt`

### 浮点数 (float)

```python
pi = 3.14159
scientific = 2.5e-3  # 0.0025

# ⚠️ 浮点数精度问题（Python 和 JS 都有）
0.1 + 0.2  # 0.30000000000000004
```

### 字符串 (str)

```python
single = 'Hello'
double = "World"
multi = """
多行
字符串
"""

# f-string（类似 JS 模板字符串）
name = "Alice"
greeting = f"Hello, {name}!"
```

### 布尔值 (bool)

```python
# ⚠️ 注意大小写！
is_valid = True   # 不是 true
is_empty = False  # 不是 false
```

**JS 对照**：

| Python | JavaScript |
|--------|------------|
| `True` | `true` |
| `False` | `false` |

### None

```python
result = None  # 类似 JS 的 null

# 检查 None
if result is None:
    print("No result")
```

**JS 对照**：

| Python | JavaScript |
|--------|------------|
| `None` | `null` |
| 无 | `undefined` |

> Python 只有 `None`，没有 `undefined`

---

## 🔄 类型转换

```python
# 转整数
int("42")       # 42
int(3.7)        # 3（截断，非四舍五入）
int("10", 2)    # 2（二进制转十进制）

# 转浮点
float("3.14")   # 3.14
float(42)       # 42.0

# 转字符串
str(42)         # "42"
str(3.14)       # "3.14"
str(True)       # "True"

# 转布尔
bool(0)         # False
bool("")        # False
bool([])        # False
bool(None)      # False
bool(1)         # True
bool("hello")   # True
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `int("42")` | `parseInt("42")` |
| `float("3.14")` | `parseFloat("3.14")` |
| `str(42)` | `String(42)` 或 `42 + ""` |
| `bool(x)` | `Boolean(x)` 或 `!!x` |

---

## 🔍 类型检查

### type() - 获取精确类型

```python
type(42)        # <class 'int'>
type(3.14)      # <class 'float'>
type("hello")   # <class 'str'>
type(True)      # <class 'bool'>
type(None)      # <class 'NoneType'>

# 类型比较
type(42) == int  # True
```

### isinstance() - 检查是否是某类型

```python
isinstance(42, int)           # True
isinstance(42, (int, float))  # True（是其中之一）
isinstance(True, int)         # True！（bool 是 int 的子类）
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `type(x)` | `typeof x` |
| `isinstance(x, Type)` | `x instanceof Type` |

**⚠️ 注意**：
- Python 的 `isinstance()` 支持继承关系
- JS 的 `typeof null` 返回 `"object"`（历史 bug）

---

## 🎭 Truthy 和 Falsy

### Python Falsy 值

```python
# 以下都是 Falsy
bool(False)    # False
bool(None)     # False
bool(0)        # False
bool(0.0)      # False
bool("")       # False
bool([])       # False（空列表）
bool({})       # False（空字典）
bool(set())    # False（空集合）
```

### JS 对照

| Python Falsy | JavaScript Falsy |
|--------------|------------------|
| `False` | `false` |
| `None` | `null`, `undefined` |
| `0`, `0.0` | `0`, `0.0`, `-0` |
| `""` | `""` |
| `[]`, `{}` | **Truthy!** |
| — | `NaN` |

> ⚠️ **重要差异**：Python 空列表/字典是 Falsy，JS 空数组/对象是 Truthy！

```python
# Python
if []:
    print("不会执行")

# JavaScript
# if ([]) { console.log("会执行！"); }
```

---

## 📦 可变 vs 不可变类型

### 不可变类型（Immutable）

```python
# int, float, str, bool, tuple

s = "hello"
s[0] = "H"  # ❌ TypeError: 'str' object does not support item assignment

# 只能创建新对象
s = "H" + s[1:]  # "Hello"
```

### 可变类型（Mutable）

```python
# list, dict, set

lst = [1, 2, 3]
lst[0] = 100  # ✅ 可以修改
```

### 为什么重要？

```python
# 默认参数陷阱
def add_item(item, lst=[]):  # ❌ 危险！
    lst.append(item)
    return lst

add_item(1)  # [1]
add_item(2)  # [1, 2]  ← 意外！共享了同一个列表

# 正确做法
def add_item(item, lst=None):  # ✅
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

---

## ✅ 本节要点

1. Python 变量无需声明关键字
2. 基本类型：`int`, `float`, `str`, `bool`, `None`
3. 类型转换：`int()`, `float()`, `str()`, `bool()`
4. 类型检查：`type()` 和 `isinstance()`
5. Python 空容器是 Falsy（与 JS 不同）
6. 理解可变 vs 不可变类型

