# 02. 继承与多态

## 🎯 本节目标

- 掌握单继承和 super()
- 理解方法重写
- 了解多继承和 MRO

---

## 📝 单继承

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "..."

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类构造器
        self.breed = breed

    def speak(self):  # 方法重写
        return f"{self.name} says Woof!"

dog = Dog("Buddy", "Golden Retriever")
print(dog.speak())  # Buddy says Woof!
print(dog.name)     # Buddy
print(dog.breed)    # Golden Retriever
```

### JS 对照

```javascript
// JavaScript
class Animal {
    constructor(name) {
        this.name = name;
    }
    speak() { return "..."; }
}

class Dog extends Animal {
    constructor(name, breed) {
        super(name);  // 必须先调用 super
        this.breed = breed;
    }
    speak() { return `${this.name} says Woof!`; }
}
```

---

## 🔄 super() 详解

```python
class Parent:
    def __init__(self, value):
        self.value = value

    def method(self):
        return f"Parent: {self.value}"

class Child(Parent):
    def __init__(self, value, extra):
        super().__init__(value)  # 调用父类方法
        self.extra = extra

    def method(self):
        parent_result = super().method()  # 调用父类方法
        return f"{parent_result}, Child: {self.extra}"

child = Child(10, 20)
print(child.method())  # Parent: 10, Child: 20
```

### ⚠️ 忘记调用 super() 的后果

```python
class Parent:
    def __init__(self):
        self.parent_attr = "parent"

class Child(Parent):
    def __init__(self):
        # 忘记 super().__init__()
        self.child_attr = "child"

child = Child()
print(child.child_attr)   # child
print(child.parent_attr)  # ❌ AttributeError
```

---

## 🔀 方法重写（Override）

```python
class Shape:
    def area(self):
        raise NotImplementedError

    def describe(self):
        return f"A shape with area {self.area()}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):  # 重写
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):  # 重写
        import math
        return math.pi * self.radius ** 2

# 多态
shapes = [Rectangle(3, 4), Circle(5)]
for shape in shapes:
    print(shape.describe())
```

---

## 👥 多继承

```python
class Flyable:
    def fly(self):
        return "Flying!"

class Swimmable:
    def swim(self):
        return "Swimming!"

class Duck(Flyable, Swimmable):
    def quack(self):
        return "Quack!"

duck = Duck()
print(duck.fly())   # Flying!
print(duck.swim())  # Swimming!
print(duck.quack()) # Quack!
```

### MRO（方法解析顺序）

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())  # B（按 MRO 顺序找到的第一个）

# 查看 MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# 或者
print(D.mro())
```

### 菱形继承（钻石问题）

```python
class A:
    def __init__(self):
        print("A.__init__")
        self.a = "A"

class B(A):
    def __init__(self):
        print("B.__init__")
        super().__init__()
        self.b = "B"

class C(A):
    def __init__(self):
        print("C.__init__")
        super().__init__()
        self.c = "C"

class D(B, C):
    def __init__(self):
        print("D.__init__")
        super().__init__()
        self.d = "D"

d = D()
# 输出：
# D.__init__
# B.__init__
# C.__init__
# A.__init__  ← A 只被调用一次！
```

> Python 使用 C3 线性化算法确保每个类只被调用一次

---

## 🔍 类型检查

```python
class Animal: pass
class Dog(Animal): pass
class Cat(Animal): pass

dog = Dog()

# isinstance：检查实例是否属于某类（包括父类）
isinstance(dog, Dog)     # True
isinstance(dog, Animal)  # True
isinstance(dog, Cat)     # False

# issubclass：检查类是否是另一个类的子类
issubclass(Dog, Animal)  # True
issubclass(Dog, Cat)     # False
issubclass(Dog, Dog)     # True
```

---

## 🎭 Mixin 类

Mixin 是用于提供额外功能的类，不应该单独实例化。

```python
class JSONMixin:
    """提供 JSON 序列化功能"""
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class LogMixin:
    """提供日志功能"""
    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")

class User(JSONMixin, LogMixin):
    def __init__(self, name, age):
        self.name = name
        self.age = age

user = User("Alice", 25)
print(user.to_json())  # {"name": "Alice", "age": 25}
user.log("Created")    # [User] Created
```

---

## ✅ 本节要点

1. `super()` 调用父类方法
2. 总是在 `__init__` 中调用 `super().__init__()`
3. 多继承使用 MRO 确定方法解析顺序
4. `isinstance()` 检查实例类型
5. `issubclass()` 检查类继承关系
6. Mixin 类提供可复用功能

