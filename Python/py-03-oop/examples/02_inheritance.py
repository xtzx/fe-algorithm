#!/usr/bin/env python3
"""
02_inheritance.py - 继承演示
"""

# =============================================================================
# 1. 单继承
# =============================================================================
print("=== 单继承 ===")


class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return "..."


class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name)  # 调用父类构造器
        self.breed = breed

    def speak(self) -> str:  # 方法重写
        return f"{self.name} says Woof!"


class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says Meow!"


dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers")
print(f"狗: {dog.speak()}")
print(f"猫: {cat.speak()}")
print(f"品种: {dog.breed}")

# =============================================================================
# 2. 多态
# =============================================================================
print("\n=== 多态 ===")

animals = [Dog("Max", "Labrador"), Cat("Luna"), Dog("Charlie", "Poodle")]
for animal in animals:
    print(animal.speak())

# =============================================================================
# 3. isinstance 和 issubclass
# =============================================================================
print("\n=== 类型检查 ===")

print(f"dog 是 Dog? {isinstance(dog, Dog)}")
print(f"dog 是 Animal? {isinstance(dog, Animal)}")
print(f"dog 是 Cat? {isinstance(dog, Cat)}")
print(f"Dog 是 Animal 子类? {issubclass(Dog, Animal)}")

# =============================================================================
# 4. 多继承
# =============================================================================
print("\n=== 多继承 ===")


class Flyable:
    def fly(self) -> str:
        return "Flying!"


class Swimmable:
    def swim(self) -> str:
        return "Swimming!"


class Duck(Flyable, Swimmable):
    def __init__(self, name: str):
        self.name = name

    def quack(self) -> str:
        return f"{self.name} says Quack!"


duck = Duck("Donald")
print(duck.quack())
print(duck.fly())
print(duck.swim())

# =============================================================================
# 5. MRO
# =============================================================================
print("\n=== MRO (方法解析顺序) ===")


class A:
    def method(self) -> str:
        return "A"


class B(A):
    def method(self) -> str:
        return "B -> " + super().method()


class C(A):
    def method(self) -> str:
        return "C -> " + super().method()


class D(B, C):
    def method(self) -> str:
        return "D -> " + super().method()


d = D()
print(f"调用链: {d.method()}")
print(f"MRO: {[cls.__name__ for cls in D.__mro__]}")

# =============================================================================
# 6. Mixin
# =============================================================================
print("\n=== Mixin ===")

import json


class JSONMixin:
    def to_json(self) -> str:
        return json.dumps(self.__dict__)


class ReprMixin:
    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class User(JSONMixin, ReprMixin):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


user = User("Alice", 25)
print(f"repr: {user}")
print(f"JSON: {user.to_json()}")

print("\n=== 运行完成 ===")

