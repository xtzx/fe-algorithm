# 01. 类基础

## 🎯 本节目标

- 掌握类的定义与实例化
- 理解 self 参数
- 区分实例属性与类属性
- 掌握三种方法类型

---

## 📝 类定义

```python
class Person:
    """人类"""

    def __init__(self, name, age):
        """构造器"""
        self.name = name  # 实例属性
        self.age = age

    def greet(self):
        """实例方法"""
        return f"Hello, I'm {self.name}"

# 创建实例
alice = Person("Alice", 25)
print(alice.greet())  # Hello, I'm Alice
```

### JS 对照

```javascript
// JavaScript
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    greet() {
        return `Hello, I'm ${this.name}`;
    }
}

const alice = new Person("Alice", 25);
```

---

## 🔑 self 参数

Python 中 `self` 必须显式声明，而 JS 中 `this` 是隐式的。

```python
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1  # 必须用 self 访问实例属性
        return self

    def get_count(self):
        return self.count

counter = Counter()
counter.increment().increment()  # 链式调用
print(counter.get_count())  # 2
```

### self vs this

| 特性 | Python self | JS this |
|------|-------------|---------|
| 声明方式 | 显式（第一个参数） | 隐式 |
| 命名 | 约定用 `self`（可改） | 固定为 `this` |
| 绑定时机 | 调用时自动绑定 | 取决于调用方式 |
| 箭头函数 | 不影响 | 继承外层 `this` |

---

## 📦 类属性 vs 实例属性

```python
class Dog:
    # 类属性：所有实例共享
    species = "Canis familiaris"
    count = 0

    def __init__(self, name):
        # 实例属性：每个实例独有
        self.name = name
        Dog.count += 1

dog1 = Dog("Buddy")
dog2 = Dog("Max")

print(Dog.species)       # Canis familiaris
print(dog1.species)      # Canis familiaris（通过实例访问）
print(Dog.count)         # 2

# 修改类属性
Dog.species = "Canis lupus"
print(dog1.species)      # Canis lupus（所有实例都变了）

# ⚠️ 通过实例赋值会创建实例属性
dog1.species = "Custom"
print(dog1.species)      # Custom（实例属性）
print(dog2.species)      # Canis lupus（类属性）
```

### ⚠️ 可变类属性陷阱

```python
# ❌ 危险
class BadClass:
    items = []  # 类属性（可变）

    def add(self, item):
        self.items.append(item)

a = BadClass()
b = BadClass()
a.add(1)
print(b.items)  # [1]  ← 所有实例共享！

# ✅ 正确
class GoodClass:
    def __init__(self):
        self.items = []  # 实例属性
```

---

## 🔧 三种方法

### 1. 实例方法

```python
class Calculator:
    def __init__(self, value=0):
        self.value = value

    def add(self, x):
        """实例方法：第一个参数是 self"""
        self.value += x
        return self
```

### 2. 类方法 @classmethod

```python
class Calculator:
    def __init__(self, value=0):
        self.value = value

    @classmethod
    def from_string(cls, s):
        """类方法：第一个参数是 cls（类本身）"""
        return cls(int(s))

# 使用类方法创建实例
calc = Calculator.from_string("42")
print(calc.value)  # 42
```

**类方法的用途**：
- 工厂方法（替代构造器）
- 修改类属性
- 在继承时返回正确的子类

### 3. 静态方法 @staticmethod

```python
class MathUtils:
    @staticmethod
    def is_even(n):
        """静态方法：不需要 self 或 cls"""
        return n % 2 == 0

# 可以通过类或实例调用
print(MathUtils.is_even(4))  # True
```

**静态方法的用途**：
- 与类相关但不需要访问实例或类属性的工具函数
- 组织代码

### 对比

| 方法类型 | 装饰器 | 第一个参数 | 用途 |
|---------|--------|-----------|------|
| 实例方法 | 无 | `self` | 操作实例 |
| 类方法 | `@classmethod` | `cls` | 工厂方法，操作类 |
| 静态方法 | `@staticmethod` | 无 | 工具函数 |

---

## 🔐 访问控制

Python 没有真正的私有属性，靠约定：

```python
class BankAccount:
    def __init__(self, balance):
        self.balance = balance       # 公开
        self._balance = balance      # 约定私有（内部使用）
        self.__balance = balance     # 名称改写（强私有）

    def get_balance(self):
        return self.__balance

account = BankAccount(100)
print(account.balance)        # 100（公开）
print(account._balance)       # 100（可访问，但不推荐）
print(account.__balance)      # ❌ AttributeError
print(account._BankAccount__balance)  # 100（名称改写后）
```

### 命名约定

| 命名 | 含义 |
|------|------|
| `name` | 公开 |
| `_name` | 内部使用（约定） |
| `__name` | 名称改写（避免子类冲突） |
| `__name__` | Python 保留（魔法方法） |

---

## 📋 文档字符串

```python
class Calculator:
    """
    简单的计算器类。

    Attributes:
        value: 当前值

    Example:
        >>> calc = Calculator(10)
        >>> calc.add(5).value
        15
    """

    def __init__(self, value: float = 0):
        """初始化计算器。

        Args:
            value: 初始值，默认为 0
        """
        self.value = value

    def add(self, x: float) -> "Calculator":
        """加法运算。

        Args:
            x: 要加的数

        Returns:
            返回自身，支持链式调用
        """
        self.value += x
        return self

# 查看文档
help(Calculator)
print(Calculator.__doc__)
```

---

## ✅ 本节要点

1. `self` 必须显式声明为第一个参数
2. 类属性被所有实例共享，可变类属性要小心
3. 实例属性在 `__init__` 中初始化
4. `@classmethod` 用于工厂方法
5. `@staticmethod` 用于工具函数
6. `_name` 约定私有，`__name` 名称改写

