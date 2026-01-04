# 03. 闭包（Closure）

## 🎯 本节目标

- 理解闭包的定义和原理
- 掌握状态保持机制
- 使用 nonlocal 关键字
- 了解常见用途

---

## 📝 什么是闭包

闭包（Closure）是指**内部函数引用了外部函数的变量**，即使外部函数已经返回，内部函数仍然可以访问这些变量。

### 基本示例

```python
def outer(x):
    # 外部函数的变量
    def inner(y):
        # 内部函数引用了外部变量 x
        return x + y
    return inner

# 创建闭包
add_5 = outer(5)
print(add_5(10))  # 15

# x 的值（5）被"记住"了
add_3 = outer(3)
print(add_3(10))  # 13
```

### 闭包的特征

1. **嵌套函数**：函数内部定义函数
2. **引用外部变量**：内部函数使用外部函数的变量
3. **返回内部函数**：外部函数返回内部函数
4. **状态保持**：外部变量被"捕获"并保持

---

## 🔒 状态保持

闭包可以"记住"外部函数的状态。

### 计数器示例

```python
def make_counter():
    count = 0  # 外部变量

    def counter():
        nonlocal count  # 声明非局部变量
        count += 1
        return count

    return counter

# 创建两个独立的计数器
c1 = make_counter()
c2 = make_counter()

print(c1())  # 1
print(c1())  # 2
print(c2())  # 1（独立的计数器）
print(c1())  # 3
```

### 配置函数示例

```python
def make_multiplier(n):
    """创建乘法器"""
    def multiplier(x):
        return x * n
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

---

## 🔑 nonlocal 关键字

当内部函数需要**修改**外部变量时，必须使用 `nonlocal`。

### 不使用 nonlocal（错误）

```python
def make_counter():
    count = 0

    def counter():
        count += 1  # ❌ UnboundLocalError
        return count

    return counter

c = make_counter()
c()  # 报错！
```

### 使用 nonlocal（正确）

```python
def make_counter():
    count = 0

    def counter():
        nonlocal count  # ✅ 声明非局部变量
        count += 1
        return count

    return counter

c = make_counter()
print(c())  # 1
print(c())  # 2
```

### nonlocal vs global

```python
# global：修改全局变量
count = 0

def increment_global():
    global count
    count += 1

# nonlocal：修改外层函数的变量
def outer():
    count = 0

    def inner():
        nonlocal count  # 不是 global！
        count += 1

    return inner
```

---

## 🏭 工厂函数

闭包常用于创建"工厂函数"。

### 日志记录器

```python
def make_logger(prefix):
    """创建带前缀的日志记录器"""
    def log(message):
        print(f"[{prefix}] {message}")
    return log

info_logger = make_logger("INFO")
error_logger = make_logger("ERROR")

info_logger("系统启动")   # [INFO] 系统启动
error_logger("发生错误")  # [ERROR] 发生错误
```

### 权限检查器

```python
def make_permission_checker(required_role):
    """创建权限检查器"""
    def check(user):
        return user.get("role") == required_role
    return check

admin_check = make_permission_checker("admin")
user_check = make_permission_checker("user")

user = {"name": "Alice", "role": "admin"}
print(admin_check(user))  # True
print(user_check(user))   # False
```

### 数据验证器

```python
def make_validator(min_val, max_val):
    """创建数值验证器"""
    def validate(value):
        if not (min_val <= value <= max_val):
            raise ValueError(f"值必须在 {min_val} 和 {max_val} 之间")
        return value
    return validate

age_validator = make_validator(0, 150)
score_validator = make_validator(0, 100)

age_validator(25)    # ✅
score_validator(85)  # ✅
# age_validator(200)  # ❌ ValueError
```

---

## ⏱️ 延迟计算

闭包可以用于延迟计算。

### 延迟求值

```python
def make_lazy(func, *args, **kwargs):
    """创建延迟执行的函数"""
    def lazy():
        return func(*args, **kwargs)
    return lazy

import time

def expensive_operation():
    time.sleep(1)
    return "结果"

# 不立即执行
lazy_result = make_lazy(expensive_operation)

# 需要时才执行
print(lazy_result())  # 等待 1 秒后返回 "结果"
```

### 缓存装饰器（简化版）

```python
def make_cached(func):
    """创建带缓存的函数"""
    cache = {}

    def cached(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return cached

@make_cached
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))  # 快速返回（有缓存）
```

---

## 🆚 Python vs JavaScript 闭包

### Python

```python
def outer(x):
    def inner(y):
        return x + y
    return inner

add_5 = outer(5)
print(add_5(10))  # 15
```

### JavaScript

```javascript
function outer(x) {
    return function inner(y) {
        return x + y;
    };
}

const add5 = outer(5);
console.log(add5(10));  // 15
```

### 主要区别

| 特性 | Python | JavaScript |
|------|--------|------------|
| 修改外部变量 | 需要 `nonlocal` | 直接修改 |
| 变量提升 | 无 | 有（var） |
| 作用域 | 函数作用域 | 函数/块作用域 |

---

## ⚠️ 常见陷阱

### 1. 循环中的闭包

```python
# ❌ 问题：所有闭包捕获最后一个值
funcs = []
for i in range(3):
    funcs.append(lambda x: x + i)

print(funcs[0](10))  # 12（所有都是 i=2）
print(funcs[1](10))  # 12
print(funcs[2](10))  # 12

# ✅ 解决 1：使用默认参数
funcs = []
for i in range(3):
    funcs.append(lambda x, i=i: x + i)

# ✅ 解决 2：使用生成器
funcs = [lambda x, i=i: x + i for i in range(3)]

# ✅ 解决 3：创建新作用域
def make_adder(n):
    return lambda x: x + n

funcs = [make_adder(i) for i in range(3)]
```

### 2. 可变对象陷阱

```python
# ⚠️ 注意：闭包引用的是对象本身
def make_appender():
    items = []  # 可变对象

    def append(item):
        items.append(item)
        return items

    return append

appender = make_appender()
print(appender(1))  # [1]
print(appender(2))  # [1, 2]
```

---

## 🎯 实际应用

### 1. 装饰器基础

```python
def timer(func):
    """计时装饰器"""
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result

    return wrapper
```

### 2. 配置管理

```python
def make_config(env):
    """根据环境创建配置"""
    configs = {
        "dev": {"debug": True, "host": "localhost"},
        "prod": {"debug": False, "host": "api.example.com"},
    }

    def get_config(key):
        return configs[env].get(key)

    return get_config

dev_config = make_config("dev")
prod_config = make_config("prod")
```

### 3. 事件处理

```python
def make_event_handler(event_type):
    """创建事件处理器"""
    handlers = []

    def register(handler):
        handlers.append(handler)

    def trigger(*args, **kwargs):
        for handler in handlers:
            handler(*args, **kwargs)

    return register, trigger

on_click, trigger_click = make_event_handler("click")
on_click(lambda: print("Clicked!"))
trigger_click()  # Clicked!
```

---

## ✅ 本节要点

1. 闭包是内部函数引用外部变量的机制
2. 闭包可以"记住"外部函数的状态
3. 修改外部变量需要使用 `nonlocal`
4. 常用于工厂函数、延迟计算、配置管理
5. 循环中创建闭包要注意变量绑定问题

