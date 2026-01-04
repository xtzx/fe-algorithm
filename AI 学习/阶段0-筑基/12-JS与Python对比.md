# 🔄 12 - JavaScript 与 Python 对比

> 专为前端开发者准备的 Python 快速入门指南

---

## 目录

1. [语法对比速查表](#1-语法对比速查表)
2. [数据结构对比](#2-数据结构对比)
3. [函数对比](#3-函数对比)
4. [类与面向对象](#4-类与面向对象)
5. [异步编程对比](#5-异步编程对比)
6. [常见陷阱](#6-常见陷阱)
7. [工具链对比](#7-工具链对比)
8. [实战练习](#8-实战练习)

---

## 1. 语法对比速查表

### 1.1 基础语法

| 特性 | JavaScript | Python |
|------|------------|--------|
| **变量声明** | `const`, `let`, `var` | 直接赋值，无关键字 |
| **常量** | `const PI = 3.14` | `PI = 3.14`（约定大写） |
| **类型系统** | 动态类型 + TypeScript | 动态类型 + Type Hints |
| **代码块** | `{ }` 花括号 | 缩进（4空格） |
| **语句结尾** | `;` 可选 | 无分号 |
| **注释** | `//` 和 `/* */` | `#` 和 `''' '''` |
| **空值** | `null`, `undefined` | `None` |
| **布尔值** | `true`, `false` | `True`, `False` |
| **打印** | `console.log()` | `print()` |

### 1.2 代码对照

```javascript
// JavaScript
const name = "Alice";
let age = 25;
const isStudent = true;

if (age >= 18) {
    console.log(`${name} is an adult`);
} else {
    console.log(`${name} is a minor`);
}

for (let i = 0; i < 5; i++) {
    console.log(i);
}
```

```python
# Python
name = "Alice"
age = 25
is_student = True

if age >= 18:
    print(f"{name} is an adult")
else:
    print(f"{name} is a minor")

for i in range(5):
    print(i)
```

### 1.3 命名规范对比

| 场景 | JavaScript | Python |
|------|------------|--------|
| **变量/函数** | `camelCase` | `snake_case` |
| **类名** | `PascalCase` | `PascalCase` |
| **常量** | `SCREAMING_SNAKE_CASE` | `SCREAMING_SNAKE_CASE` |
| **私有成员** | `#privateField` / `_private` | `_private` / `__private` |
| **文件名** | `camelCase.js` / `kebab-case.js` | `snake_case.py` |

```javascript
// JavaScript 风格
const userName = "alice";
function getUserInfo() { }
class UserProfile { }
const MAX_RETRY_COUNT = 3;
```

```python
# Python 风格
user_name = "alice"
def get_user_info(): pass
class UserProfile: pass
MAX_RETRY_COUNT = 3
```

### 1.4 类型注解对比

```typescript
// TypeScript
function greet(name: string, age: number): string {
    return `Hello, ${name}! You are ${age} years old.`;
}

interface User {
    id: number;
    name: string;
    email?: string;  // 可选
}

const users: User[] = [];
```

```python
# Python Type Hints
def greet(name: str, age: int) -> str:
    return f"Hello, {name}! You are {age} years old."

from typing import Optional, List
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: Optional[str] = None  # 可选

users: List[User] = []

# Python 3.9+ 可以直接用
users: list[User] = []
```

---

## 2. 数据结构对比

### 2.1 Array vs List

```javascript
// JavaScript Array
const fruits = ["apple", "banana", "cherry"];

// 访问
console.log(fruits[0]);        // "apple"
console.log(fruits.at(-1));    // "cherry" (ES2022)

// 添加
fruits.push("date");           // 末尾添加
fruits.unshift("avocado");     // 开头添加

// 删除
fruits.pop();                  // 删除末尾
fruits.shift();                // 删除开头
fruits.splice(1, 1);           // 删除指定位置

// 长度
console.log(fruits.length);

// 遍历
fruits.forEach((fruit, index) => {
    console.log(`${index}: ${fruit}`);
});

// 映射
const upper = fruits.map(f => f.toUpperCase());

// 过滤
const longNames = fruits.filter(f => f.length > 5);

// 查找
const found = fruits.find(f => f.startsWith("b"));
const index = fruits.findIndex(f => f === "banana");

// 判断
const hasApple = fruits.includes("apple");
const allLong = fruits.every(f => f.length > 3);
const someLong = fruits.some(f => f.length > 5);

// 归约
const total = [1, 2, 3].reduce((sum, n) => sum + n, 0);

// 排序
fruits.sort();                           // 原地排序
fruits.sort((a, b) => a.localeCompare(b));

// 切片（返回新数组）
const sliced = fruits.slice(1, 3);

// 展开
const moreFruits = [...fruits, "elderberry"];
```

```python
# Python List
fruits = ["apple", "banana", "cherry"]

# 访问
print(fruits[0])         # "apple"
print(fruits[-1])        # "cherry" (Python 原生支持负索引！)

# 添加
fruits.append("date")           # 末尾添加
fruits.insert(0, "avocado")     # 开头添加

# 删除
fruits.pop()                    # 删除末尾
fruits.pop(0)                   # 删除开头（无 shift）
del fruits[1]                   # 删除指定位置
fruits.remove("banana")         # 删除指定值

# 长度
print(len(fruits))              # len() 是函数，不是属性！

# 遍历
for fruit in fruits:
    print(fruit)

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# 映射（列表推导式）
upper = [f.upper() for f in fruits]

# 过滤
long_names = [f for f in fruits if len(f) > 5]

# 查找
found = next((f for f in fruits if f.startswith("b")), None)
try:
    index = fruits.index("banana")
except ValueError:
    index = -1

# 判断
has_apple = "apple" in fruits
all_long = all(len(f) > 3 for f in fruits)
some_long = any(len(f) > 5 for f in fruits)

# 归约
from functools import reduce
total = reduce(lambda sum, n: sum + n, [1, 2, 3], 0)
# 或者直接用 sum()
total = sum([1, 2, 3])

# 排序
fruits.sort()                                    # 原地排序
fruits.sort(key=lambda x: x.lower())             # 自定义
sorted_fruits = sorted(fruits)                   # 返回新列表

# 切片
sliced = fruits[1:3]

# 展开
more_fruits = [*fruits, "elderberry"]
```

### 2.2 ⚠️ 前端开发者易踩坑

```python
# 🔴 坑 1：len() 是函数，不是属性
arr = [1, 2, 3]
# print(arr.length)  # ❌ AttributeError
print(len(arr))      # ✅ 3

# 🔴 坑 2：Python 没有 forEach
# arr.forEach(...)   # ❌ 没有这个方法
for item in arr:     # ✅ 用 for 循环
    print(item)

# 🔴 坑 3：map/filter 返回迭代器，不是列表
result = map(lambda x: x * 2, [1, 2, 3])
print(result)        # <map object at 0x...>
print(list(result))  # [2, 4, 6]

# ✅ 推荐用列表推导式
result = [x * 2 for x in [1, 2, 3]]

# 🔴 坑 4：负索引是合法的！
arr = [1, 2, 3]
print(arr[-1])  # 3（最后一个）
print(arr[-2])  # 2（倒数第二个）
# JS 中 arr[-1] 是 undefined
```

### 2.3 Object vs Dict

```javascript
// JavaScript Object
const person = {
    name: "Alice",
    age: 25,
    "special-key": "value"  // 特殊键名需要引号
};

// 访问
console.log(person.name);           // 点语法
console.log(person["age"]);         // 括号语法
console.log(person["special-key"]); // 特殊键名必须用括号

// 添加/修改
person.city = "NYC";
person["country"] = "USA";

// 删除
delete person.city;

// 检查键存在
console.log("name" in person);                    // true
console.log(person.hasOwnProperty("name"));       // true

// 获取键/值
console.log(Object.keys(person));    // ["name", "age", ...]
console.log(Object.values(person));  // ["Alice", 25, ...]
console.log(Object.entries(person)); // [["name", "Alice"], ...]

// 遍历
for (const [key, value] of Object.entries(person)) {
    console.log(`${key}: ${value}`);
}

// 展开
const extended = { ...person, email: "a@b.com" };

// 解构
const { name, age } = person;

// 可选链
console.log(person?.address?.city);  // undefined（不报错）
```

```python
# Python Dict
person = {
    "name": "Alice",
    "age": 25,
    "special-key": "value"  # 所有键都需要引号（除非是变量）
}

# 访问
print(person["name"])              # ✅ 键必须是字符串
# print(person.name)               # ❌ 不支持点语法！
print(person.get("age"))           # ✅ 更安全的访问
print(person.get("height", 170))   # 提供默认值

# 添加/修改
person["city"] = "NYC"

# 删除
del person["city"]
person.pop("country", None)        # 安全删除

# 检查键存在
print("name" in person)            # True

# 获取键/值
print(person.keys())               # dict_keys(['name', 'age', ...])
print(list(person.keys()))         # ['name', 'age', ...]
print(person.values())
print(person.items())              # [('name', 'Alice'), ...]

# 遍历
for key, value in person.items():
    print(f"{key}: {value}")

# 展开（合并）
extended = {**person, "email": "a@b.com"}

# 解构（Python 没有直接解构，但可以用 .values()）
name, age = person["name"], person["age"]

# 或者用 operator.itemgetter
from operator import itemgetter
name, age = itemgetter("name", "age")(person)

# 安全访问嵌套（Python 没有可选链，需要 get 链式调用）
address = person.get("address", {}).get("city")
```

### 2.4 ⚠️ Object vs Dict 关键差异

```python
# 🔴 坑 1：Dict 不支持点语法
person = {"name": "Alice"}
# print(person.name)    # ❌ AttributeError
print(person["name"])   # ✅ "Alice"

# 💡 如果想用点语法，用 dataclass 或 namedtuple
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

p = Person(name="Alice", age=25)
print(p.name)  # ✅ "Alice"

# 🔴 坑 2：访问不存在的键
person = {"name": "Alice"}
# print(person["age"])  # ❌ KeyError
print(person.get("age"))        # ✅ None
print(person.get("age", 0))     # ✅ 0

# 🔴 坑 3：Dict 的键可以是任何不可变类型
d = {
    (1, 2): "tuple key",      # 元组可以做键
    42: "int key",            # 数字可以做键
    # [1, 2]: "list key"      # ❌ 列表不行（可变）
}
```

### 2.5 Set 对比

```javascript
// JavaScript Set
const set = new Set([1, 2, 3, 2, 1]);
console.log(set);        // Set(3) {1, 2, 3}

set.add(4);
set.delete(1);
console.log(set.has(2)); // true
console.log(set.size);   // 3

// 转数组
const arr = [...set];
const arr2 = Array.from(set);

// 集合运算（ES2025+ 或手动实现）
const a = new Set([1, 2, 3]);
const b = new Set([2, 3, 4]);
// 并集
const union = new Set([...a, ...b]);
// 交集
const intersection = new Set([...a].filter(x => b.has(x)));
// 差集
const difference = new Set([...a].filter(x => !b.has(x)));
```

```python
# Python Set
s = {1, 2, 3, 2, 1}
print(s)             # {1, 2, 3}

s.add(4)
s.remove(1)          # 不存在会报错
s.discard(1)         # 不存在不报错
print(2 in s)        # True
print(len(s))        # 3

# 转列表
arr = list(s)

# 集合运算（原生支持！）
a = {1, 2, 3}
b = {2, 3, 4}
print(a | b)  # {1, 2, 3, 4} 并集
print(a & b)  # {2, 3} 交集
print(a - b)  # {1} 差集
print(a ^ b)  # {1, 4} 对称差集

# 🔴 注意：空集合
empty_set = set()    # ✅ 正确
# empty_set = {}     # ❌ 这是空字典！
```

### 2.6 解构赋值对比

```javascript
// JavaScript 解构
// 数组解构
const [a, b, ...rest] = [1, 2, 3, 4, 5];
console.log(a, b, rest);  // 1 2 [3, 4, 5]

// 对象解构
const { name, age, city = "Unknown" } = person;

// 交换变量
let x = 1, y = 2;
[x, y] = [y, x];

// 嵌套解构
const { address: { city } } = { address: { city: "NYC" } };
```

```python
# Python 解构（称为"解包"）
# 列表/元组解包
a, b, *rest = [1, 2, 3, 4, 5]
print(a, b, rest)  # 1 2 [3, 4, 5]

# 字典没有直接解构语法
# 需要显式获取
name, age = person["name"], person["age"]

# 或者用这种方式
name, age = person.values()  # 但顺序依赖于插入顺序

# 交换变量（超简洁！）
x, y = 1, 2
x, y = y, x

# 函数返回多值
def get_point():
    return 10, 20

x, y = get_point()
```

---

## 3. 函数对比

### 3.1 函数定义

```javascript
// JavaScript
// 函数声明（会提升）
function greet(name) {
    return `Hello, ${name}!`;
}

// 函数表达式
const greet = function(name) {
    return `Hello, ${name}!`;
};

// 箭头函数
const greet = (name) => `Hello, ${name}!`;
const greet = name => `Hello, ${name}!`;  // 单参数可省略括号
const add = (a, b) => a + b;

// 多行箭头函数
const calculate = (a, b) => {
    const sum = a + b;
    return sum * 2;
};
```

```python
# Python
# 函数定义（不会提升！）
def greet(name):
    return f"Hello, {name}!"

# lambda 表达式（仅限单表达式）
greet = lambda name: f"Hello, {name}!"
add = lambda a, b: a + b

# 🔴 lambda 不能有多行！
# calculate = lambda a, b:
#     sum = a + b    # ❌ SyntaxError
#     return sum * 2

# 多行必须用 def
def calculate(a, b):
    total = a + b
    return total * 2
```

### 3.2 参数对比

```javascript
// JavaScript
// 默认参数
function greet(name, greeting = "Hello") {
    return `${greeting}, ${name}!`;
}

// 剩余参数
function sum(...numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}

// 解构参数
function createUser({ name, age, city = "Unknown" }) {
    return { name, age, city };
}
createUser({ name: "Alice", age: 25 });

// 参数对象
function config(options = {}) {
    const { timeout = 1000, retries = 3 } = options;
    // ...
}
```

```python
# Python
# 默认参数
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# *args（位置参数）
def sum_all(*numbers):
    return sum(numbers)

sum_all(1, 2, 3, 4)  # 10

# **kwargs（关键字参数）
def create_user(**kwargs):
    name = kwargs.get("name")
    age = kwargs.get("age")
    city = kwargs.get("city", "Unknown")
    return {"name": name, "age": age, "city": city}

create_user(name="Alice", age=25)

# 混合使用
def func(a, b, *args, **kwargs):
    print(f"a={a}, b={b}, args={args}, kwargs={kwargs}")

func(1, 2, 3, 4, x=5, y=6)
# a=1, b=2, args=(3, 4), kwargs={'x': 5, 'y': 6}

# 强制关键字参数（* 后面的必须用关键字）
def greet(name, *, greeting="Hello"):
    return f"{greeting}, {name}!"

# greet("Alice", "Hi")        # ❌ TypeError
greet("Alice", greeting="Hi") # ✅
```

### 3.3 ⚠️ 默认参数陷阱

```python
# 🔴 严重陷阱：可变默认参数
def add_item(item, items=[]):  # ❌ 危险！
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] 😱 不是 ['b']！

# ✅ 正确做法
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['b'] ✅
```

### 3.4 高阶函数

```javascript
// JavaScript
// 回调函数
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);
const sum = numbers.reduce((acc, n) => acc + n, 0);

// 函数作为返回值
function multiplier(factor) {
    return (x) => x * factor;
}
const double = multiplier(2);
console.log(double(5));  // 10

// 立即执行函数
(function() {
    console.log("IIFE");
})();

// 柯里化
const add = a => b => a + b;
console.log(add(1)(2));  // 3
```

```python
# Python
# 高阶函数
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda n: n * 2, numbers))
evens = list(filter(lambda n: n % 2 == 0, numbers))
from functools import reduce
total = reduce(lambda acc, n: acc + n, numbers, 0)

# ✅ 推荐：列表推导式（更 Pythonic）
doubled = [n * 2 for n in numbers]
evens = [n for n in numbers if n % 2 == 0]
total = sum(numbers)

# 函数作为返回值（闭包）
def multiplier(factor):
    def inner(x):
        return x * factor
    return inner

double = multiplier(2)
print(double(5))  # 10

# 柯里化
def add(a):
    def inner(b):
        return a + b
    return inner

print(add(1)(2))  # 3

# 或用 functools.partial
from functools import partial
def add(a, b):
    return a + b

add_one = partial(add, 1)
print(add_one(2))  # 3
```

---

## 4. 类与面向对象

### 4.1 类定义

```javascript
// JavaScript (ES6+)
class Animal {
    // 私有字段（ES2022）
    #secretId = Math.random();

    // 静态属性
    static kingdom = "Animalia";

    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    // 方法
    speak() {
        console.log(`${this.name} makes a sound`);
    }

    // Getter
    get info() {
        return `${this.name}, ${this.age} years old`;
    }

    // Setter
    set info(value) {
        [this.name, this.age] = value.split(", ");
    }

    // 静态方法
    static create(name) {
        return new Animal(name, 0);
    }
}

// 使用
const dog = new Animal("Buddy", 3);
dog.speak();
console.log(dog.info);
console.log(Animal.kingdom);
```

```python
# Python
class Animal:
    # 类属性（相当于静态属性）
    kingdom = "Animalia"

    def __init__(self, name, age):
        self.name = name        # 实例属性
        self.age = age
        self._secret_id = id(self)  # 约定私有（单下划线）
        self.__real_private = 1     # 名称改写（双下划线）

    # 方法（第一个参数必须是 self）
    def speak(self):
        print(f"{self.name} makes a sound")

    # 属性（Getter）
    @property
    def info(self):
        return f"{self.name}, {self.age} years old"

    # Setter
    @info.setter
    def info(self, value):
        self.name, self.age = value.split(", ")
        self.age = int(self.age)

    # 静态方法（不需要 self）
    @staticmethod
    def create(name):
        return Animal(name, 0)

    # 类方法（第一个参数是 cls）
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["age"])

# 使用
dog = Animal("Buddy", 3)  # 不需要 new！
dog.speak()
print(dog.info)
print(Animal.kingdom)
```

### 4.2 继承

```javascript
// JavaScript
class Dog extends Animal {
    constructor(name, age, breed) {
        super(name, age);  // 调用父类构造函数
        this.breed = breed;
    }

    speak() {
        super.speak();  // 调用父类方法
        console.log(`${this.name} barks!`);
    }
}

const buddy = new Dog("Buddy", 3, "Golden Retriever");
buddy.speak();
// Buddy makes a sound
// Buddy barks!

console.log(buddy instanceof Dog);    // true
console.log(buddy instanceof Animal); // true
```

```python
# Python
class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age)  # 调用父类
        self.breed = breed

    def speak(self):
        super().speak()  # 调用父类方法
        print(f"{self.name} barks!")

buddy = Dog("Buddy", 3, "Golden Retriever")
buddy.speak()
# Buddy makes a sound
# Buddy barks!

print(isinstance(buddy, Dog))     # True
print(isinstance(buddy, Animal))  # True

# Python 支持多重继承！
class A:
    def method(self):
        print("A")

class B:
    def method(self):
        print("B")

class C(A, B):  # 多重继承
    pass

c = C()
c.method()  # "A"（按 MRO 顺序）
print(C.__mro__)  # 查看方法解析顺序
```

### 4.3 魔术方法（Dunder Methods）

```python
# Python 特有的魔术方法
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 字符串表示
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

    # 运算符重载
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    # 比较
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # 长度
    def __len__(self):
        return 2

    # 索引访问
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Index out of range")

    # 迭代
    def __iter__(self):
        yield self.x
        yield self.y

# 使用
v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1 + v2)       # (4, 6)
print(v1 * 3)        # (3, 6)
print(v1 == v2)      # False
print(len(v1))       # 2
print(v1[0])         # 1
print(list(v1))      # [1, 2]
```

### 4.4 数据类（dataclass）

```python
# Python 3.7+ dataclass（类似 TypeScript interface）
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class User:
    id: int
    name: str
    email: str
    tags: List[str] = field(default_factory=list)
    is_active: bool = True

    # 可以添加方法
    def greet(self):
        return f"Hello, {self.name}!"

# 自动生成 __init__, __repr__, __eq__ 等
user = User(id=1, name="Alice", email="alice@example.com")
print(user)
# User(id=1, name='Alice', email='alice@example.com', tags=[], is_active=True)

# 不可变版本
@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
# p.x = 3.0  # ❌ FrozenInstanceError
```

---

## 5. 异步编程对比

### 5.1 Promise vs asyncio

```javascript
// JavaScript - Promise
function fetchUser(id) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (id > 0) {
                resolve({ id, name: "Alice" });
            } else {
                reject(new Error("Invalid ID"));
            }
        }, 1000);
    });
}

// 使用 then/catch
fetchUser(1)
    .then(user => console.log(user))
    .catch(err => console.error(err));

// 链式调用
fetchUser(1)
    .then(user => fetchPosts(user.id))
    .then(posts => console.log(posts))
    .catch(err => console.error(err));
```

```python
# Python - asyncio
import asyncio

async def fetch_user(user_id):
    await asyncio.sleep(1)  # 模拟网络请求
    if user_id > 0:
        return {"id": user_id, "name": "Alice"}
    else:
        raise ValueError("Invalid ID")

# 运行异步函数
async def main():
    try:
        user = await fetch_user(1)
        print(user)
    except ValueError as e:
        print(f"Error: {e}")

# 启动事件循环
asyncio.run(main())
```

### 5.2 async/await 对比

```javascript
// JavaScript
async function getUserData(userId) {
    try {
        const user = await fetchUser(userId);
        const posts = await fetchPosts(user.id);
        const comments = await fetchComments(posts[0].id);
        return { user, posts, comments };
    } catch (error) {
        console.error("Error:", error);
        throw error;
    }
}

// 并行执行
async function getAllData() {
    const [users, products] = await Promise.all([
        fetchUsers(),
        fetchProducts()
    ]);
    return { users, products };
}

// Promise.race
const result = await Promise.race([
    fetchFromServer1(),
    fetchFromServer2()
]);
```

```python
# Python
async def get_user_data(user_id):
    try:
        user = await fetch_user(user_id)
        posts = await fetch_posts(user["id"])
        comments = await fetch_comments(posts[0]["id"])
        return {"user": user, "posts": posts, "comments": comments}
    except Exception as e:
        print(f"Error: {e}")
        raise

# 并行执行
async def get_all_data():
    users, products = await asyncio.gather(
        fetch_users(),
        fetch_products()
    )
    return {"users": users, "products": products}

# 竞速
async def race_example():
    done, pending = await asyncio.wait(
        [fetch_from_server1(), fetch_from_server2()],
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    return done.pop().result()
```

### 5.3 ⚠️ 异步关键差异

```python
# 🔴 坑 1：必须用 asyncio.run() 启动
async def main():
    result = await some_async_function()

# main()  # ❌ 返回协程对象，不会执行
asyncio.run(main())  # ✅

# 🔴 坑 2：不能在普通函数中用 await
def normal_function():
    # result = await async_function()  # ❌ SyntaxError
    pass

# 🔴 坑 3：同步和异步不能混用
# requests 是同步库
import requests
# response = await requests.get(url)  # ❌ 不行

# 要用异步 HTTP 库
import aiohttp
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.json()

# 🔴 坑 4：Jupyter 中的异步
# Jupyter 已经在运行事件循环，直接 await 即可
# await some_async_function()  # ✅ 在 Jupyter 中直接用
```

---

## 6. 常见陷阱

### 6.1 作用域差异

```javascript
// JavaScript - 块级作用域（let/const）
if (true) {
    let x = 1;
    const y = 2;
}
// console.log(x);  // ❌ ReferenceError

// var 会提升（不推荐）
if (true) {
    var z = 3;
}
console.log(z);  // 3
```

```python
# Python - 没有块级作用域！
if True:
    x = 1

print(x)  # 1 ✅ 可以访问！

# 函数才有作用域
def func():
    y = 2

# print(y)  # ❌ NameError

# 🔴 循环变量泄漏
for i in range(5):
    pass
print(i)  # 4（循环变量泄漏到外部！）
```

### 6.2 this vs self

```javascript
// JavaScript - this 很复杂
const obj = {
    name: "Alice",
    greet: function() {
        console.log(this.name);  // this 依赖调用方式
    },
    greetArrow: () => {
        console.log(this.name);  // 箭头函数的 this 绑定外层
    }
};

obj.greet();           // "Alice"
const fn = obj.greet;
fn();                  // undefined（this 丢失）

// 需要 bind
const boundFn = obj.greet.bind(obj);
boundFn();             // "Alice"
```

```python
# Python - self 必须显式传递，但很简单
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(self.name)  # self 必须写

person = Person("Alice")
person.greet()         # "Alice"

fn = person.greet      # 方法绑定了实例
fn()                   # "Alice" ✅ 不会丢失

# 类方法 vs 静态方法
class Demo:
    @staticmethod
    def static_method():
        pass  # 没有 self

    @classmethod
    def class_method(cls):
        pass  # cls 是类本身
```

### 6.3 真值判断

```javascript
// JavaScript 的假值
// false, 0, -0, "", null, undefined, NaN

console.log(Boolean(""));     // false
console.log(Boolean(0));      // false
console.log(Boolean([]));     // true ⚠️ 空数组是真值！
console.log(Boolean({}));     // true ⚠️ 空对象是真值！

// 宽松相等的坑
console.log([] == false);     // true 😱
console.log([] == ![]);       // true 😱😱
```

```python
# Python 的假值
# False, None, 0, 0.0, "", [], {}, set()

print(bool(""))        # False
print(bool(0))         # False
print(bool([]))        # False ⚠️ 空列表是假值！
print(bool({}))        # False ⚠️ 空字典是假值！

# 比较一致性（Python 的 == 很正常）
print([] == False)     # False（类型不同）

# 常见用法
items = []
if not items:          # ✅ Pythonic 写法
    print("Empty")

if items:              # ✅ 有内容才执行
    process(items)
```

### 6.4 is vs ==

```python
# 🔴 这是 Python 特有的坑

# == 比较值
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True（值相等）

# is 比较身份（内存地址）
print(a is b)  # False（不是同一个对象）

# 特殊情况：小整数缓存
x = 256
y = 256
print(x is y)  # True（Python 缓存了 -5 到 256）

x = 257
y = 257
print(x is y)  # False（超出缓存范围）

# ✅ 判断 None 要用 is
value = None
if value is None:      # ✅ 正确
    pass
if value == None:      # ⚠️ 能工作，但不推荐
    pass
```

### 6.5 浅拷贝 vs 深拷贝

```javascript
// JavaScript
const original = { a: 1, nested: { b: 2 } };

// 浅拷贝
const shallow = { ...original };
shallow.nested.b = 999;
console.log(original.nested.b);  // 999 😱 原对象也改了

// 深拷贝
const deep = JSON.parse(JSON.stringify(original));  // 有限制
const deep2 = structuredClone(original);             // 现代方法
```

```python
# Python
import copy

original = {"a": 1, "nested": {"b": 2}}

# 浅拷贝
shallow = original.copy()  # 或 dict(original)
shallow["nested"]["b"] = 999
print(original["nested"]["b"])  # 999 😱 原对象也改了

# 深拷贝
deep = copy.deepcopy(original)
deep["nested"]["b"] = 999
print(original["nested"]["b"])  # 2 ✅ 原对象不受影响

# 列表的拷贝
arr = [[1, 2], [3, 4]]
shallow = arr.copy()     # 或 arr[:]
deep = copy.deepcopy(arr)
```

### 6.6 可变默认参数（Python 独有坑）

```python
# 🔴🔴🔴 Python 最经典的坑

def add_item(item, items=[]):  # ❌ 危险！
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] 😱😱😱

# 原因：默认参数在函数定义时创建一次，之后复用
# JavaScript 每次调用都会创建新的默认值

# ✅ 正确写法
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

---

## 7. 工具链对比

### 7.1 包管理器

| 功能 | JavaScript | Python |
|------|------------|--------|
| **包管理器** | npm / yarn / pnpm | pip / conda / uv |
| **依赖文件** | `package.json` | `requirements.txt` / `pyproject.toml` |
| **锁文件** | `package-lock.json` | `requirements.txt` / `poetry.lock` |
| **安装依赖** | `npm install` | `pip install -r requirements.txt` |
| **安装单个包** | `npm install lodash` | `pip install numpy` |
| **开发依赖** | `npm install -D jest` | `pip install pytest` |
| **全局安装** | `npm install -g xxx` | `pip install xxx`（通常不推荐） |
| **运行脚本** | `npm run build` | 直接运行 / `python -m module` |

```bash
# JavaScript
npm init -y
npm install express
npm install -D typescript
npm run dev

# Python
pip install numpy pandas matplotlib
pip freeze > requirements.txt
pip install -r requirements.txt

# 推荐用 uv（更快）
pip install uv
uv pip install numpy
```

### 7.2 代码质量工具

| 功能 | JavaScript | Python |
|------|------------|--------|
| **Linter** | ESLint | pylint / flake8 / **ruff** |
| **Formatter** | Prettier | **black** / autopep8 |
| **类型检查** | TypeScript | mypy / pyright |
| **测试框架** | Jest / Vitest | **pytest** / unittest |

```bash
# 推荐 Python 工具组合
pip install ruff black pytest mypy

# ruff（超快的 linter）
ruff check .
ruff format .

# black（格式化）
black .

# pytest（测试）
pytest

# mypy（类型检查）
mypy .
```

### 7.3 项目结构对比

```
# JavaScript/TypeScript 项目
my-project/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts
│   └── utils/
├── tests/
├── dist/
└── node_modules/

# Python 项目
my-project/
├── pyproject.toml  # 或 requirements.txt
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── utils.py
├── tests/
│   └── test_utils.py
└── venv/  # 虚拟环境（不提交到 git）
```

---

## 8. 实战练习

### 练习 1：数据处理

将以下 JavaScript 代码转换为 Python：

```javascript
// JavaScript
const users = [
    { name: "Alice", age: 25, active: true },
    { name: "Bob", age: 30, active: false },
    { name: "Charlie", age: 35, active: true }
];

// 1. 过滤活跃用户
const activeUsers = users.filter(u => u.active);

// 2. 获取姓名列表
const names = users.map(u => u.name);

// 3. 计算平均年龄
const avgAge = users.reduce((sum, u) => sum + u.age, 0) / users.length;

// 4. 按年龄排序
const sorted = [...users].sort((a, b) => a.age - b.age);

console.log({ activeUsers, names, avgAge, sorted });
```

<details>
<summary>参考答案</summary>

```python
# Python
users = [
    {"name": "Alice", "age": 25, "active": True},
    {"name": "Bob", "age": 30, "active": False},
    {"name": "Charlie", "age": 35, "active": True}
]

# 1. 过滤活跃用户
active_users = [u for u in users if u["active"]]

# 2. 获取姓名列表
names = [u["name"] for u in users]

# 3. 计算平均年龄
avg_age = sum(u["age"] for u in users) / len(users)

# 4. 按年龄排序
sorted_users = sorted(users, key=lambda u: u["age"])

print({
    "active_users": active_users,
    "names": names,
    "avg_age": avg_age,
    "sorted": sorted_users
})
```

</details>

### 练习 2：异步请求

将以下 JavaScript 代码转换为 Python：

```javascript
// JavaScript
async function fetchUserAndPosts(userId) {
    try {
        const userResponse = await fetch(`/api/users/${userId}`);
        const user = await userResponse.json();

        const postsResponse = await fetch(`/api/users/${userId}/posts`);
        const posts = await postsResponse.json();

        return { user, posts };
    } catch (error) {
        console.error("Failed to fetch:", error);
        throw error;
    }
}
```

<details>
<summary>参考答案</summary>

```python
# Python（使用 aiohttp）
import aiohttp
import asyncio

async def fetch_user_and_posts(user_id):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"/api/users/{user_id}") as response:
                user = await response.json()

            async with session.get(f"/api/users/{user_id}/posts") as response:
                posts = await response.json()

        return {"user": user, "posts": posts}
    except Exception as e:
        print(f"Failed to fetch: {e}")
        raise

# 运行
# asyncio.run(fetch_user_and_posts(1))
```

</details>

### 练习 3：类实现

将以下 TypeScript 代码转换为 Python：

```typescript
// TypeScript
interface TodoItem {
    id: number;
    title: string;
    completed: boolean;
}

class TodoList {
    private items: TodoItem[] = [];
    private nextId: number = 1;

    add(title: string): TodoItem {
        const item: TodoItem = {
            id: this.nextId++,
            title,
            completed: false
        };
        this.items.push(item);
        return item;
    }

    toggle(id: number): void {
        const item = this.items.find(i => i.id === id);
        if (item) {
            item.completed = !item.completed;
        }
    }

    getAll(): TodoItem[] {
        return [...this.items];
    }

    getCompleted(): TodoItem[] {
        return this.items.filter(i => i.completed);
    }
}
```

<details>
<summary>参考答案</summary>

```python
# Python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TodoItem:
    id: int
    title: str
    completed: bool = False

class TodoList:
    def __init__(self):
        self._items: List[TodoItem] = []
        self._next_id: int = 1

    def add(self, title: str) -> TodoItem:
        item = TodoItem(
            id=self._next_id,
            title=title,
            completed=False
        )
        self._next_id += 1
        self._items.append(item)
        return item

    def toggle(self, id: int) -> None:
        item = next((i for i in self._items if i.id == id), None)
        if item:
            item.completed = not item.completed

    def get_all(self) -> List[TodoItem]:
        return self._items.copy()

    def get_completed(self) -> List[TodoItem]:
        return [i for i in self._items if i.completed]

# 使用
todo = TodoList()
todo.add("Learn Python")
todo.add("Build a project")
todo.toggle(1)
print(todo.get_completed())
```

</details>

---

## 📚 速查表

### 常用操作对照

| 操作 | JavaScript | Python |
|------|------------|--------|
| 打印 | `console.log()` | `print()` |
| 长度 | `arr.length` | `len(arr)` |
| 类型 | `typeof x` | `type(x)` |
| 范围 | `Array.from({length: 5}, (_, i) => i)` | `range(5)` |
| 字符串格式化 | `` `Hello ${name}` `` | `f"Hello {name}"` |
| 判断存在 | `arr.includes(x)` | `x in arr` |
| 映射 | `arr.map(fn)` | `[fn(x) for x in arr]` |
| 过滤 | `arr.filter(fn)` | `[x for x in arr if fn(x)]` |
| 排序 | `arr.sort((a, b) => a - b)` | `arr.sort()` 或 `sorted(arr)` |
| 空判断 | `if (arr.length === 0)` | `if not arr:` |
| 三元 | `a ? b : c` | `b if a else c` |
| 空合并 | `a ?? b` | `a if a is not None else b` |
| 可选链 | `obj?.prop` | `obj.get("prop")` |

---

## ➡️ 下一步

学完本节后，继续学习 [13-数学符号速查.md](./13-数学符号速查.md)

