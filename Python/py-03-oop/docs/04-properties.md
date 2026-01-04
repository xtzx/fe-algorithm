# 04. 属性与描述符

## 🎯 本节目标

- 掌握 @property 装饰器
- 实现属性验证
- 了解描述符协议

---

## 📝 @property 基础

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """getter：获取半径"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """setter：设置半径"""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @radius.deleter
    def radius(self):
        """deleter：删除半径"""
        del self._radius

circle = Circle(5)
print(circle.radius)    # 5（调用 getter）
circle.radius = 10      # 调用 setter
print(circle.radius)    # 10
# circle.radius = -1    # ❌ ValueError
```

### JS 对照

```javascript
// JavaScript
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

## 📊 计算属性

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        """只读计算属性"""
        return self.width * self.height

    @property
    def perimeter(self):
        """只读计算属性"""
        return 2 * (self.width + self.height)

rect = Rectangle(3, 4)
print(rect.area)       # 12
print(rect.perimeter)  # 14
# rect.area = 20       # ❌ AttributeError: can't set
```

---

## ✅ 属性验证

```python
class Person:
    def __init__(self, name, age):
        self.name = name  # 触发 setter
        self.age = age

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not value or not isinstance(value, str):
            raise ValueError("Name must be non-empty string")
        self._name = value.strip()

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Age must be non-negative integer")
        self._age = value

# 验证生效
person = Person("Alice", 25)
# Person("", 25)    # ❌ ValueError
# Person("Bob", -1) # ❌ ValueError
```

---

## 🔒 只读属性

```python
class Configuration:
    def __init__(self, settings):
        self._settings = settings.copy()

    @property
    def settings(self):
        """只读：返回副本"""
        return self._settings.copy()

config = Configuration({"debug": True})
settings = config.settings
settings["debug"] = False
print(config.settings)  # {"debug": True}（未被修改）
```

---

## 🎭 描述符协议

描述符是 `@property` 的底层机制。

### 数据描述符

```python
class Positive:
    """确保值为正数的描述符"""

    def __set_name__(self, owner, name):
        self.name = name
        self.storage_name = f"__{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.storage_name, None)

    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        setattr(obj, self.storage_name, value)

class Product:
    price = Positive()
    quantity = Positive()

    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

product = Product("Widget", 10, 5)
print(product.price)     # 10
# product.price = -1     # ❌ ValueError
```

### 描述符协议方法

| 方法 | 用途 |
|------|------|
| `__get__(self, obj, type)` | 获取属性值 |
| `__set__(self, obj, value)` | 设置属性值 |
| `__delete__(self, obj)` | 删除属性 |
| `__set_name__(self, owner, name)` | 获取属性名 |

---

## 🔧 property 的其他用法

### 使用 property() 函数

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius

    def get_fahrenheit(self):
        return self._celsius * 9/5 + 32

    def set_fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

    fahrenheit = property(get_fahrenheit, set_fahrenheit)

temp = Temperature(100)
print(temp.fahrenheit)  # 212.0
temp.fahrenheit = 32
print(temp._celsius)    # 0.0
```

### 缓存属性

```python
from functools import cached_property

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    @cached_property
    def expensive_calculation(self):
        """只计算一次"""
        print("Calculating...")
        return sum(self.data) / len(self.data)

analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print(analyzer.expensive_calculation)  # Calculating... 3.0
print(analyzer.expensive_calculation)  # 3.0（不再计算）
```

---

## ✅ 本节要点

1. `@property` 创建 getter
2. `@xxx.setter` 创建 setter
3. 不定义 setter 则为只读属性
4. 描述符是 property 的底层实现
5. `@cached_property` 缓存计算结果

