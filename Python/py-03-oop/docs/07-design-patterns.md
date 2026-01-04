# 07. 设计模式

## 🎯 本节目标

- 实现单例模式
- 实现工厂模式
- 理解 Mixin 类

---

## 🔒 单例模式

确保类只有一个实例。

### 方式 1：使用 __new__

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value=None):
        # 注意：每次调用都会执行
        if not hasattr(self, 'initialized'):
            self.value = value
            self.initialized = True

s1 = Singleton("first")
s2 = Singleton("second")
print(s1 is s2)      # True
print(s1.value)      # first
```

### 方式 2：使用装饰器

```python
def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Database:
    def __init__(self, url):
        self.url = url
        print(f"Connecting to {url}")

db1 = Database("mysql://localhost")  # Connecting to mysql://localhost
db2 = Database("postgres://localhost")  # 不会打印
print(db1 is db2)  # True
```

### 方式 3：使用模块

```python
# config.py
class _Config:
    def __init__(self):
        self.settings = {}

config = _Config()  # 模块级单例

# 其他文件
from config import config
```

---

## 🏭 工厂模式

### 简单工厂

```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create(animal_type: str) -> Animal:
        animals = {
            "dog": Dog,
            "cat": Cat,
        }
        if animal_type not in animals:
            raise ValueError(f"Unknown animal: {animal_type}")
        return animals[animal_type]()

dog = AnimalFactory.create("dog")
print(dog.speak())  # Woof!
```

### 工厂方法

```python
from abc import ABC, abstractmethod

class Document(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

class PDFDocument(Document):
    def render(self):
        return "Rendering PDF"

class HTMLDocument(Document):
    def render(self):
        return "Rendering HTML"

class DocumentCreator(ABC):
    @abstractmethod
    def create_document(self) -> Document:
        pass

    def open(self) -> str:
        doc = self.create_document()
        return doc.render()

class PDFCreator(DocumentCreator):
    def create_document(self):
        return PDFDocument()

class HTMLCreator(DocumentCreator):
    def create_document(self):
        return HTMLDocument()

creator = PDFCreator()
print(creator.open())  # Rendering PDF
```

### 使用 classmethod 作为工厂

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
        """工厂方法"""
        return cls(name, email, datetime.now())

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """从字典创建"""
        return cls(
            name=data["name"],
            email=data["email"],
            created_at=datetime.fromisoformat(data["created_at"])
        )

user = User.create("Alice", "alice@example.com")
```

---

## 🧩 Mixin 类

Mixin 提供可复用的功能，不应该单独实例化。

```python
import json

class JSONMixin:
    """提供 JSON 序列化功能"""

    def to_json(self) -> str:
        return json.dumps(self._to_dict())

    def _to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

class ComparableMixin:
    """提供比较功能"""

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

class ReprMixin:
    """提供 repr 功能"""

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

class User(JSONMixin, ComparableMixin, ReprMixin):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

user = User("Alice", 25)
print(user)              # User(name='Alice', age=25)
print(user.to_json())    # {"name": "Alice", "age": 25}
print(user == User("Alice", 25))  # True
```

---

## 🎨 策略模式

```python
from abc import ABC, abstractmethod
from typing import List

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

class BubbleSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        result = data.copy()
        n = len(result)
        for i in range(n):
            for j in range(0, n-i-1):
                if result[j] > result[j+1]:
                    result[j], result[j+1] = result[j+1], result[j]
        return result

class QuickSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        return sorted(data)

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy

    def sort(self, data: List[int]) -> List[int]:
        return self._strategy.sort(data)

data = [3, 1, 4, 1, 5, 9, 2, 6]
sorter = Sorter(BubbleSort())
print(sorter.sort(data))

sorter.set_strategy(QuickSort())
print(sorter.sort(data))
```

---

## 🔌 观察者模式

```python
from abc import ABC, abstractmethod
from typing import List

class Observer(ABC):
    @abstractmethod
    def update(self, message: str):
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer):
        self._observers.append(observer)

    def detach(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self, message: str):
        for observer in self._observers:
            observer.update(message)

class EmailNotifier(Observer):
    def update(self, message: str):
        print(f"Email: {message}")

class SlackNotifier(Observer):
    def update(self, message: str):
        print(f"Slack: {message}")

# 使用
subject = Subject()
subject.attach(EmailNotifier())
subject.attach(SlackNotifier())
subject.notify("New order received!")
# Email: New order received!
# Slack: New order received!
```

---

## ✅ 本节要点

1. 单例：`__new__`、装饰器、模块
2. 工厂：`@classmethod` 作为工厂方法
3. Mixin：提供可复用功能的类
4. 策略：运行时选择算法
5. 观察者：一对多依赖通知

