# 10. 面试高频问题

> 10 个 Python 基础面试高频问题

---

## 1. Python 2 和 Python 3 的主要区别？

<details>
<summary>参考答案</summary>

| 特性 | Python 2 | Python 3 |
|------|----------|----------|
| `print` | 语句 `print "hi"` | 函数 `print("hi")` |
| 整数除法 | `3/2 = 1` | `3/2 = 1.5` |
| Unicode | `u"文字"` | 默认 Unicode |
| `range()` | 返回列表 | 返回迭代器 |
| `input()` | `raw_input()` | `input()` |
| 异常语法 | `except E, e:` | `except E as e:` |

**关键点**：Python 2 已于 2020 年 1 月 1 日停止维护，所有新项目都应使用 Python 3。

</details>

---

## 2. `is` 和 `==` 的区别？

<details>
<summary>参考答案</summary>

- **`==`**：比较**值**是否相等（调用 `__eq__` 方法）
- **`is`**：比较**身份**是否相同（是否是同一个对象，比较 `id()`）

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b  # True（值相同）
a is b  # False（不同对象）
a is c  # True（同一对象）

# 小整数缓存
x = 256
y = 256
x is y  # True（-5 到 256 被缓存）

x = 257
y = 257
x is y  # 可能 False
```

**最佳实践**：
- 比较值用 `==`
- 比较 `None` 用 `is`：`if x is None`

</details>

---

## 3. 可变对象和不可变对象的区别？

<details>
<summary>参考答案</summary>

**不可变对象**（Immutable）：
- `int`, `float`, `str`, `tuple`, `frozenset`, `bool`
- 创建后不能修改
- 作为字典键安全

**可变对象**（Mutable）：
- `list`, `dict`, `set`
- 可以原地修改
- 不能作为字典键

```python
# 不可变
s = "hello"
s[0] = "H"  # ❌ TypeError

# 可变
lst = [1, 2, 3]
lst[0] = 100  # ✅

# 函数参数陷阱
def add_item(item, lst=[]):  # ❌ 可变默认参数
    lst.append(item)
    return lst

add_item(1)  # [1]
add_item(2)  # [1, 2]（共享同一个列表！）
```

</details>

---

## 4. Python 为什么没有 switch（3.10 之前）？

<details>
<summary>参考答案</summary>

**历史原因**：Python 之父 Guido 认为：
1. `if/elif/else` 足够清晰
2. 字典映射可以替代
3. 避免 C 风格 switch 的 fallthrough 问题

**替代方案**：

```python
# 字典映射
def handle_command(cmd):
    handlers = {
        "start": start_handler,
        "stop": stop_handler,
        "restart": restart_handler,
    }
    return handlers.get(cmd, default_handler)()
```

**Python 3.10+ 有 match-case**：

```python
match command:
    case "start":
        start()
    case "stop":
        stop()
    case _:
        default()
```

</details>

---

## 5. Python 的作用域规则是什么？

<details>
<summary>参考答案</summary>

**LEGB 规则**（按查找顺序）：

1. **L**ocal：函数内部局部变量
2. **E**nclosing：外层函数的局部变量
3. **G**lobal：模块级全局变量
4. **B**uilt-in：内置变量（如 `len`, `print`）

```python
x = "global"  # G

def outer():
    x = "enclosing"  # E

    def inner():
        x = "local"  # L
        print(x)  # local

    inner()

# 修改外层变量
def counter():
    count = 0  # E

    def increment():
        nonlocal count  # 声明使用外层变量
        count += 1
        return count

    return increment
```

</details>

---

## 6. 如何交换两个变量的值？

<details>
<summary>参考答案</summary>

```python
# Python 最简洁的方式
a, b = b, a

# 原理：元组解包
# 等价于：
# temp = (b, a)
# a = temp[0]
# b = temp[1]
```

**JS/其他语言对比**：

```javascript
// JS ES6+
[a, b] = [b, a];

// 传统方式
let temp = a;
a = b;
b = temp;
```

</details>

---

## 7. `range()` 和 `xrange()` 的区别？

<details>
<summary>参考答案</summary>

**Python 2**：
- `range()` 返回**列表**（一次性生成所有元素）
- `xrange()` 返回**迭代器**（惰性生成）

**Python 3**：
- `range()` 返回**迭代器**（类似 Python 2 的 `xrange`）
- `xrange()` 被移除

```python
# Python 3
r = range(1000000)  # 不占用大量内存
type(r)  # <class 'range'>

# 需要列表时
lst = list(range(10))
```

</details>

---

## 8. Python 字符串是可变的吗？

<details>
<summary>参考答案</summary>

**不可变**（Immutable）。

```python
s = "hello"
s[0] = "H"  # ❌ TypeError: 'str' object does not support item assignment

# 只能创建新字符串
s = "H" + s[1:]  # "Hello"
s = s.replace("h", "H")
```

**为什么不可变？**
1. 性能：字符串哈希可缓存
2. 安全：可作为字典键
3. 线程安全

</details>

---

## 9. f-string、format() 和 % 的区别？

<details>
<summary>参考答案</summary>

| 方式 | 语法 | Python 版本 | 性能 |
|------|------|------------|------|
| f-string | `f"Hello {name}"` | 3.6+ | 最快 |
| format() | `"Hello {}".format(name)` | 2.6+ | 中等 |
| % 格式化 | `"Hello %s" % name` | 所有 | 较慢 |

```python
name = "Alice"
age = 25

# f-string（推荐）
f"Name: {name}, Age: {age}"

# format()
"Name: {}, Age: {}".format(name, age)
"Name: {n}, Age: {a}".format(n=name, a=age)

# % 格式化（老式）
"Name: %s, Age: %d" % (name, age)
```

**推荐优先级**：f-string > format() > %

</details>

---

## 10. Python 的 True/False 与 JS 的 truthy/falsy 有何不同？

<details>
<summary>参考答案</summary>

**Python Falsy 值**：
- `False`
- `None`
- `0`, `0.0`, `0j`
- `""`, `()`, `[]`, `{}`, `set()`, `frozenset()`

**JavaScript Falsy 值**：
- `false`
- `null`, `undefined`
- `0`, `-0`, `0n`
- `""`
- `NaN`

**关键区别**：
- **空容器**：Python Falsy，JS Truthy！

```python
# Python
if []:
    print("不会执行")

if {}:
    print("不会执行")
```

```javascript
// JavaScript
if ([]) {
    console.log("会执行！"); // 空数组是 truthy
}

if ({}) {
    console.log("会执行！"); // 空对象是 truthy
}
```

</details>

---

## 📝 更多面试准备

### 常见陷阱

1. **可变默认参数**：`def f(lst=[])`
2. **is vs ==**：对象比较用 `==`
3. **整数除法**：`/` 返回 float
4. **字符串不可变**
5. **缩进错误**

### 代码题常考

1. 两数之和
2. 反转字符串
3. 回文判断
4. 斐波那契数列
5. FizzBuzz
6. 合并两个有序列表
7. 实现栈/队列
8. 二分查找

### 概念题常考

1. GIL（全局解释器锁）
2. 装饰器原理
3. 迭代器 vs 生成器
4. 深拷贝 vs 浅拷贝
5. `*args` 和 `**kwargs`

