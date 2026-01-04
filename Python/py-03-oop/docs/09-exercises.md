# 09. 练习题

> 20 道 OOP 练习题

---

## 📝 类基础（6 道）

### 1. 银行账户类

**题目**：实现 BankAccount 类，支持存款、取款、查询余额。

<details>
<summary>答案</summary>

```python
class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self._balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient balance")
        self._balance -= amount

    @property
    def balance(self) -> float:
        return self._balance

# 测试
account = BankAccount("Alice", 100)
account.deposit(50)
account.withdraw(30)
print(account.balance)  # 120
```

</details>

---

### 2. 学生类与班级类

**题目**：实现 Student 和 Classroom 类，Classroom 包含多个学生，能计算平均分。

<details>
<summary>答案</summary>

```python
class Student:
    def __init__(self, name: str, score: float):
        self.name = name
        self.score = score

class Classroom:
    def __init__(self, name: str):
        self.name = name
        self.students: list[Student] = []

    def add_student(self, student: Student) -> None:
        self.students.append(student)

    def average_score(self) -> float:
        if not self.students:
            return 0.0
        return sum(s.score for s in self.students) / len(self.students)

    def top_student(self) -> Student | None:
        if not self.students:
            return None
        return max(self.students, key=lambda s: s.score)

classroom = Classroom("Python 101")
classroom.add_student(Student("Alice", 90))
classroom.add_student(Student("Bob", 85))
print(classroom.average_score())  # 87.5
```

</details>

---

### 3. 计数器类

**题目**：实现一个计数器类，支持链式调用。

<details>
<summary>答案</summary>

```python
class Counter:
    def __init__(self, initial: int = 0):
        self.value = initial

    def increment(self, n: int = 1) -> "Counter":
        self.value += n
        return self

    def decrement(self, n: int = 1) -> "Counter":
        self.value -= n
        return self

    def reset(self) -> "Counter":
        self.value = 0
        return self

counter = Counter()
counter.increment().increment(5).decrement(2)
print(counter.value)  # 4
```

</details>

---

### 4. 工厂方法

**题目**：为 User 类添加多个创建方式。

<details>
<summary>答案</summary>

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class User:
    name: str
    email: str
    created_at: datetime

    @classmethod
    def create(cls, name: str, email: str) -> "User":
        return cls(name, email, datetime.now())

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            name=data["name"],
            email=data["email"],
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )

    @classmethod
    def guest(cls) -> "User":
        return cls("Guest", "guest@example.com", datetime.now())

user = User.create("Alice", "alice@example.com")
guest = User.guest()
```

</details>

---

### 5. 温度转换器

**题目**：实现 Temperature 类，支持摄氏度和华氏度互转。

<details>
<summary>答案</summary>

```python
class Temperature:
    def __init__(self, celsius: float = 0):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        self._celsius = (value - 32) * 5/9

    @classmethod
    def from_fahrenheit(cls, f: float) -> "Temperature":
        return cls((f - 32) * 5/9)

temp = Temperature(100)
print(temp.fahrenheit)  # 212.0
temp.fahrenheit = 32
print(temp.celsius)     # 0.0
```

</details>

---

### 6. 单例模式

**题目**：实现 Logger 单例类。

<details>
<summary>答案</summary>

```python
class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, message: str):
        self.logs.append(message)
        print(f"[LOG] {message}")

    def get_logs(self) -> list[str]:
        return self.logs.copy()

logger1 = Logger()
logger2 = Logger()
print(logger1 is logger2)  # True
```

</details>

---

## ⚡ 魔法方法（6 道）

### 7. 向量类

**题目**：实现 Vector 类，支持加法、乘法运算。

<details>
<summary>答案</summary>

```python
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __abs__(self):
        return (self.x**2 + self.y**2) ** 0.5

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)    # Vector(4, 6)
print(v1 * 3)     # Vector(3, 6)
print(abs(v2))    # 5.0
```

</details>

---

### 8. 自定义列表

**题目**：实现 MyList 类，支持索引和遍历。

<details>
<summary>答案</summary>

```python
class MyList:
    def __init__(self, items=None):
        self._items = list(items) if items else []

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __delitem__(self, index):
        del self._items[index]

    def __contains__(self, item):
        return item in self._items

    def __iter__(self):
        return iter(self._items)

    def __repr__(self):
        return f"MyList({self._items})"

lst = MyList([1, 2, 3])
print(len(lst))     # 3
print(lst[0])       # 1
print(2 in lst)     # True
for item in lst:
    print(item)
```

</details>

---

### 9. 上下文管理器

**题目**：实现 Timer 上下文管理器，测量代码执行时间。

<details>
<summary>答案</summary>

```python
import time

class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed:.4f} seconds")
        return False

with Timer("Sleep test"):
    time.sleep(0.1)
# Sleep test: 0.1001 seconds
```

</details>

---

### 10. 可调用类

**题目**：实现一个记忆函数调用次数的 Counter 类。

<details>
<summary>答案</summary>

```python
class CallCounter:
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)

@CallCounter
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # Hello, Alice!
print(greet("Bob"))    # Hello, Bob!
print(greet.calls)     # 2
```

</details>

---

### 11. 比较方法

**题目**：实现 Version 类，支持版本号比较。

<details>
<summary>答案</summary>

```python
from functools import total_ordering

@total_ordering
class Version:
    def __init__(self, version: str):
        parts = version.split(".")
        self.major = int(parts[0]) if len(parts) > 0 else 0
        self.minor = int(parts[1]) if len(parts) > 1 else 0
        self.patch = int(parts[2]) if len(parts) > 2 else 0

    def __repr__(self):
        return f"Version({self.major}.{self.minor}.{self.patch})"

    def __eq__(self, other):
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

v1 = Version("1.2.3")
v2 = Version("1.3.0")
print(v1 < v2)   # True
print(v1 == Version("1.2.3"))  # True
```

</details>

---

### 12. 哈希方法

**题目**：实现 Point 类，可作为字典键。

<details>
<summary>答案</summary>

```python
class Point:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

points = {Point(0, 0): "origin", Point(1, 0): "right"}
print(points[Point(0, 0)])  # origin
```

</details>

---

## 🔀 继承（4 道）

### 13. 形状继承

**题目**：实现 Shape 抽象类和 Rectangle、Circle 子类。

<details>
<summary>答案</summary>

```python
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius
```

</details>

---

### 14. Mixin 类

**题目**：实现 JSONMixin 和 ComparableMixin。

<details>
<summary>答案</summary>

```python
import json

class JSONMixin:
    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))

class ComparableMixin:
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class User(JSONMixin, ComparableMixin):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

user = User("Alice", 25)
json_str = user.to_json()
user2 = User.from_json(json_str)
print(user == user2)  # True
```

</details>

---

### 15. MRO 理解

**题目**：预测以下代码的输出。

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B" + super().method()

class C(A):
    def method(self):
        return "C" + super().method()

class D(B, C):
    def method(self):
        return "D" + super().method()

print(D().method())
print(D.__mro__)
```

<details>
<summary>答案</summary>

```
DBCA
(<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
```

MRO 顺序：D → B → C → A → object

</details>

---

### 16. 子类扩展

**题目**：扩展 list 类，添加 average 方法。

<details>
<summary>答案</summary>

```python
class NumberList(list):
    def average(self) -> float:
        if not self:
            return 0.0
        return sum(self) / len(self)

    def median(self) -> float:
        if not self:
            return 0.0
        sorted_list = sorted(self)
        n = len(sorted_list)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_list[mid - 1] + sorted_list[mid]) / 2
        return sorted_list[mid]

nums = NumberList([1, 2, 3, 4, 5])
print(nums.average())  # 3.0
print(nums.median())   # 3
nums.append(6)
print(nums.average())  # 3.5
```

</details>

---

## 📋 dataclass（4 道）

### 17. 基本 dataclass

**题目**：用 dataclass 实现 Product 类。

<details>
<summary>答案</summary>

```python
from dataclasses import dataclass, field

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        return self.price * self.quantity

product = Product("Widget", 9.99, 10)
print(product)
print(product.total_value)  # 99.9
```

</details>

---

### 18. 不可变 dataclass

**题目**：实现不可变的 Point 类。

<details>
<summary>答案</summary>

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2) ** 0.5

p1 = Point(0, 0)
p2 = Point(3, 4)
print(p1.distance_to(p2))  # 5.0
# p1.x = 1  # ❌ FrozenInstanceError

# 可作为字典键
points = {p1: "origin"}
```

</details>

---

### 19. post_init

**题目**：使用 __post_init__ 自动计算字段。

<details>
<summary>答案</summary>

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    perimeter: float = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)

rect = Rectangle(3, 4)
print(rect.area)       # 12
print(rect.perimeter)  # 14
```

</details>

---

### 20. dataclass 继承

**题目**：实现 dataclass 继承。

<details>
<summary>答案</summary>

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Employee(Person):
    employee_id: str
    department: str
    salary: float = 0.0

employee = Employee(
    name="Alice",
    age=30,
    employee_id="E001",
    department="Engineering",
    salary=100000
)
print(employee)
```

</details>

