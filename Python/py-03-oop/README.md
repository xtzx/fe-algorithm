# P03: 面向对象编程

> 面向 JS/TS 资深工程师的 Python OOP 教程

## 🎯 学完后能做

- ✅ 设计和实现 Python 类
- ✅ 理解魔法方法和协议
- ✅ 使用 dataclass 简化数据类

---

## 🚀 快速开始

```bash
cd examples
python3 01_class_basics.py
```

---

## 📚 目录结构

```
py-03-oop/
├── README.md
├── docs/
│   ├── 01-class-basics.md        # 类基础
│   ├── 02-inheritance.md         # 继承
│   ├── 03-magic-methods.md       # 魔法方法
│   ├── 04-properties.md          # 属性
│   ├── 05-abstract-protocol.md   # 抽象类与协议
│   ├── 06-dataclass.md           # dataclass
│   ├── 07-design-patterns.md     # 设计模式
│   ├── 08-js-comparison.md       # JS 对照
│   ├── 09-exercises.md           # 练习题
│   ├── 10-interview-questions.md # 面试题
│   ├── 11-descriptors.md         # 描述符协议 ⭐
│   ├── 12-metaclass.md           # 元类 ⭐
│   └── 13-dynamic-attrs.md       # 动态属性 ⭐
├── examples/
├── exercises/
├── project/
│   └── poker_game/
└── scripts/
```

---

## ⚡ Python class vs JavaScript class

| 特性 | Python | JavaScript |
|------|--------|------------|
| 构造器 | `__init__` | `constructor` |
| 实例引用 | `self`（显式） | `this`（隐式） |
| 私有属性 | `_name` / `__name` | `#name` |
| 类方法 | `@classmethod` | `static`（部分） |
| 静态方法 | `@staticmethod` | `static` |
| getter/setter | `@property` | `get` / `set` |
| 多继承 | ✅ 支持 | ❌ 不支持 |
| 抽象类 | `abc.ABC` | 无原生支持 |

---

## 🔥 核心概念速查

### 类定义

```python
class Person:
    # 类属性
    species = "Human"

    def __init__(self, name, age):
        # 实例属性
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, I'm {self.name}"

    @classmethod
    def create_anonymous(cls):
        return cls("Anonymous", 0)

    @staticmethod
    def is_adult(age):
        return age >= 18
```

### 继承

```python
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
```

### 魔法方法

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __len__(self):
        return 2
```

### Property

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value
```

### dataclass

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    label: str = ""
```

---

## ⚠️ 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| **可变类属性** | 被所有实例共享 | 在 `__init__` 中初始化 |
| **忘记 super()** | 父类未初始化 | 总是调用 `super().__init__()` |
| **双下划线改写** | `__name` 变成 `_Class__name` | 用单下划线 `_name` |
| **实现 `__eq__` 后** | `__hash__` 被设为 None | 同时实现 `__hash__` |

---

## 📖 学习路径

### 基础篇

1. [类基础](docs/01-class-basics.md)
2. [继承](docs/02-inheritance.md)
3. [魔法方法](docs/03-magic-methods.md)
4. [属性](docs/04-properties.md)
5. [抽象类与协议](docs/05-abstract-protocol.md)
6. [dataclass](docs/06-dataclass.md)
7. [设计模式](docs/07-design-patterns.md)
8. [JS 对照](docs/08-js-comparison.md)

### 进阶篇：元编程

9. [描述符协议](docs/11-descriptors.md) ⭐ - 属性访问的底层机制
10. [元类](docs/12-metaclass.md) ⭐ - 类的类，控制类创建
11. [动态属性](docs/13-dynamic-attrs.md) ⭐ - __getattr__、动态类创建

### 练习

12. [练习题](docs/09-exercises.md)
13. [面试题](docs/10-interview-questions.md)

---

## 🛠️ 小项目：扑克牌游戏

```bash
python3 project/poker_game/main.py
```

实现 Card、Deck 类，支持洗牌、发牌、排序。

