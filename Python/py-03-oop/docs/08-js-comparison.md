# 08. Python OOP vs JavaScript

## 📊 核心对照表

| 特性 | Python | JavaScript |
|------|--------|------------|
| 构造器 | `__init__` | `constructor` |
| 实例引用 | `self`（显式） | `this`（隐式） |
| 类方法 | `@classmethod` | `static`（部分） |
| 静态方法 | `@staticmethod` | `static` |
| 私有属性 | `_name` / `__name` | `#name` |
| getter/setter | `@property` | `get` / `set` |
| 多继承 | ✅ 支持 | ❌ 不支持 |
| 抽象类 | `abc.ABC` | 无原生支持 |
| 接口 | `Protocol` | TypeScript `interface` |

---

## 🏗️ 类定义对比

### Python

```python
class Person:
    species = "Human"  # 类属性

    def __init__(self, name, age):
        self.name = name  # 实例属性
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

### JavaScript

```javascript
class Person {
    static species = "Human";  // 类属性

    constructor(name, age) {
        this.name = name;  // 实例属性
        this.age = age;
    }

    greet() {
        return `Hello, I'm ${this.name}`;
    }

    static createAnonymous() {
        return new Person("Anonymous", 0);
    }

    static isAdult(age) {
        return age >= 18;
    }
}
```

---

## 🔑 self vs this

| 特性 | Python `self` | JavaScript `this` |
|------|---------------|-------------------|
| 声明 | 显式（第一个参数） | 隐式 |
| 命名 | 约定（可改） | 固定 |
| 绑定 | 调用时自动绑定 | 取决于调用方式 |
| 箭头函数 | 不影响 | 继承外层 `this` |

```python
# Python - self 必须显式声明
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
```

```javascript
// JavaScript - this 隐式存在
class Counter {
    constructor() {
        this.count = 0;
    }

    increment() {
        this.count++;
    }
}
```

---

## 🔐 私有属性

### Python

```python
class BankAccount:
    def __init__(self, balance):
        self.balance = balance      # 公开
        self._internal = "private"  # 约定私有
        self.__secret = "secret"    # 名称改写
```

### JavaScript

```javascript
class BankAccount {
    #secret;  // 真正的私有（ES2022+）

    constructor(balance) {
        this.balance = balance;
        this._internal = "private";  // 约定私有
        this.#secret = "secret";
    }
}
```

---

## 🔧 getter/setter

### Python

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
            raise ValueError("Negative")
        self._radius = value
```

### JavaScript

```javascript
class Circle {
    #radius;

    constructor(radius) {
        this.#radius = radius;
    }

    get radius() {
        return this.#radius;
    }

    set radius(value) {
        if (value < 0) throw new Error("Negative");
        this.#radius = value;
    }
}
```

---

## 🔀 继承

### Python

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed
```

### JavaScript

```javascript
class Animal {
    constructor(name) {
        this.name = name;
    }
}

class Dog extends Animal {
    constructor(name, breed) {
        super(name);  // 必须先调用
        this.breed = breed;
    }
}
```

### 多继承

```python
# Python 支持多继承
class Flyable:
    def fly(self): pass

class Swimmable:
    def swim(self): pass

class Duck(Flyable, Swimmable):
    pass
```

```javascript
// JavaScript 不支持多继承，用 Mixin 模拟
const Flyable = {
    fly() { console.log("Flying"); }
};

const Swimmable = {
    swim() { console.log("Swimming"); }
};

class Duck {}
Object.assign(Duck.prototype, Flyable, Swimmable);
```

---

## ⚡ 魔法方法 vs 特殊方法

| Python | JavaScript | 用途 |
|--------|------------|------|
| `__str__` | `toString()` | 字符串表示 |
| `__repr__` | 无 | 开发者表示 |
| `__len__` | `length` 属性 | 长度 |
| `__iter__` | `[Symbol.iterator]` | 迭代 |
| `__getitem__` | `Proxy` | 索引访问 |
| `__call__` | 函数本身 | 可调用 |
| `__eq__` | 无 | 相等比较 |
| `__add__` | 无 | 运算符重载 |

```python
# Python 运算符重载
class Vector:
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v = Vector(1, 2) + Vector(3, 4)
```

```javascript
// JavaScript 不支持运算符重载
class Vector {
    add(other) {
        return new Vector(this.x + other.x, this.y + other.y);
    }
}

const v = new Vector(1, 2).add(new Vector(3, 4));
```

---

## 📋 数据类

### Python dataclass

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# 自动生成 __init__, __repr__, __eq__
```

### TypeScript interface + class

```typescript
interface IPoint {
    x: number;
    y: number;
}

class Point implements IPoint {
    constructor(public x: number, public y: number) {}
}
```

---

## 🎯 接口/协议

### Python Protocol

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str: ...

# 不需要显式继承
class Circle:
    def draw(self) -> str:
        return "Circle"
```

### TypeScript interface

```typescript
interface Drawable {
    draw(): string;
}

class Circle implements Drawable {
    draw(): string {
        return "Circle";
    }
}
```

---

## ✅ 关键差异总结

1. **self 显式声明** - Python 需要显式写 self
2. **多继承** - Python 支持，JS 不支持
3. **运算符重载** - Python 支持，JS 不支持
4. **私有属性** - Python 靠约定，JS 用 #
5. **抽象类** - Python 有 ABC，JS 无原生支持
6. **类型检查** - Python 运行时，TS 编译时

