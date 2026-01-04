# 02. collections - 高级容器

## 🎯 本节目标

- 掌握 Counter、defaultdict、deque
- 了解 namedtuple、ChainMap
- 理解何时使用高级容器

---

## 📊 Counter - 计数器

```python
from collections import Counter

# 创建
c = Counter(["a", "b", "a", "c", "a", "b"])
print(c)  # Counter({'a': 3, 'b': 2, 'c': 1})

c = Counter("hello")
print(c)  # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})

c = Counter(a=3, b=2)
print(c)  # Counter({'a': 3, 'b': 2})
```

### 常用方法

```python
from collections import Counter

c = Counter(["a", "b", "a", "c", "a", "b"])

# 最常见的 N 个
print(c.most_common(2))  # [('a', 3), ('b', 2)]

# 访问计数（不存在返回 0）
print(c["a"])  # 3
print(c["x"])  # 0

# 更新计数
c.update(["a", "d"])
print(c)  # Counter({'a': 4, 'b': 2, 'c': 1, 'd': 1})

# 减少计数
c.subtract(["a", "a"])
print(c)  # Counter({'b': 2, 'a': 2, 'c': 1, 'd': 1})

# 获取所有元素
print(list(c.elements()))  # ['a', 'a', 'b', 'b', 'c', 'd']

# 总计数
print(c.total())  # 6 (Python 3.10+)
```

### Counter 运算

```python
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)

print(c1 + c2)  # Counter({'a': 4, 'b': 3})
print(c1 - c2)  # Counter({'a': 2})（只保留正数）
print(c1 & c2)  # Counter({'a': 1, 'b': 1})（取最小）
print(c1 | c2)  # Counter({'a': 3, 'b': 2})（取最大）
```

---

## 📖 defaultdict - 默认值字典

```python
from collections import defaultdict

# 默认值为 int（0）
counter = defaultdict(int)
counter["a"] += 1
counter["b"] += 1
counter["a"] += 1
print(counter)  # defaultdict(<class 'int'>, {'a': 2, 'b': 1})

# 默认值为 list
groups = defaultdict(list)
groups["A"].append("Alice")
groups["B"].append("Bob")
groups["A"].append("Anna")
print(groups)
# defaultdict(<class 'list'>, {'A': ['Alice', 'Anna'], 'B': ['Bob']})

# 默认值为 set
unique = defaultdict(set)
unique["colors"].add("red")
unique["colors"].add("blue")
unique["colors"].add("red")
print(unique)  # defaultdict(<class 'set'>, {'colors': {'red', 'blue'}})
```

### 自定义默认值

```python
from collections import defaultdict

# 使用 lambda
d = defaultdict(lambda: "N/A")
print(d["missing"])  # N/A

# 使用嵌套 defaultdict
nested = defaultdict(lambda: defaultdict(int))
nested["a"]["x"] += 1
nested["a"]["y"] += 2
print(nested["a"]["x"])  # 1
```

### vs 普通 dict

```python
# 普通 dict
d = {}
for word in words:
    if word not in d:
        d[word] = 0
    d[word] += 1

# 或使用 setdefault
d = {}
for word in words:
    d.setdefault(word, 0)
    d[word] += 1

# defaultdict 更简洁
d = defaultdict(int)
for word in words:
    d[word] += 1
```

---

## 🔄 deque - 双端队列

```python
from collections import deque

# 创建
d = deque()
d = deque([1, 2, 3])
d = deque([1, 2, 3], maxlen=5)  # 限制长度

# 右端操作
d.append(4)      # 右端添加
d.pop()          # 右端弹出

# 左端操作
d.appendleft(0)  # 左端添加
d.popleft()      # 左端弹出

# 扩展
d.extend([5, 6])      # 右端扩展
d.extendleft([0, -1]) # 左端扩展（注意顺序反转）
```

### 旋转

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

d.rotate(2)   # 右旋
print(d)      # deque([4, 5, 1, 2, 3])

d.rotate(-2)  # 左旋
print(d)      # deque([1, 2, 3, 4, 5])
```

### 固定长度队列

```python
from collections import deque

# 只保留最近 5 个
recent = deque(maxlen=5)
for i in range(10):
    recent.append(i)
print(recent)  # deque([5, 6, 7, 8, 9], maxlen=5)
```

### deque vs list

| 操作 | deque | list |
|------|-------|------|
| 右端添加 | O(1) | O(1) |
| 右端弹出 | O(1) | O(1) |
| 左端添加 | O(1) | O(n) |
| 左端弹出 | O(1) | O(n) |
| 随机访问 | O(n) | O(1) |

**使用 deque**：需要两端操作
**使用 list**：需要随机访问

---

## 🏷️ namedtuple - 命名元组

```python
from collections import namedtuple

# 定义
Point = namedtuple('Point', ['x', 'y'])
# 或
Point = namedtuple('Point', 'x y')

# 创建实例
p = Point(10, 20)
p = Point(x=10, y=20)

# 访问
print(p.x, p.y)    # 10 20
print(p[0], p[1])  # 10 20（也支持索引）

# 解包
x, y = p
```

### 高级用法

```python
from collections import namedtuple

Person = namedtuple('Person', ['name', 'age', 'city'], defaults=['Unknown'])

# 使用默认值
p1 = Person("Alice", 25)
print(p1)  # Person(name='Alice', age=25, city='Unknown')

# 转为字典
print(p1._asdict())  # {'name': 'Alice', 'age': 25, 'city': 'Unknown'}

# 替换字段
p2 = p1._replace(age=26)
print(p2)  # Person(name='Alice', age=26, city='Unknown')
```

### vs typing.NamedTuple

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
    label: str = ""

p = Point(10, 20, "origin")
```

---

## 🔗 ChainMap - 字典链

```python
from collections import ChainMap

# 创建
defaults = {"color": "red", "size": "medium"}
user_prefs = {"color": "blue"}

config = ChainMap(user_prefs, defaults)
print(config["color"])  # blue（优先使用第一个）
print(config["size"])   # medium（回退到第二个）
```

### 实际应用

```python
from collections import ChainMap
import os

# 配置优先级：命令行 > 环境变量 > 默认
cli_args = {"debug": True}
env_vars = dict(os.environ)
defaults = {"debug": False, "log_level": "INFO"}

config = ChainMap(cli_args, env_vars, defaults)
```

---

## 📋 OrderedDict - 有序字典

**注意**：Python 3.7+ 普通 dict 已保序，OrderedDict 主要用于：

```python
from collections import OrderedDict

# 移动到末尾
d = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
d.move_to_end("a")
print(d)  # OrderedDict([('b', 2), ('c', 3), ('a', 1)])

# 移动到开头
d.move_to_end("c", last=False)
print(d)  # OrderedDict([('c', 3), ('b', 2), ('a', 1)])

# 弹出最后/最前
d.popitem(last=True)   # 弹出最后
d.popitem(last=False)  # 弹出最前
```

---

## ✅ 本节要点

1. `Counter` 计数，`most_common()` 获取最常见元素
2. `defaultdict` 自动初始化默认值
3. `deque` 双端 O(1) 操作，`maxlen` 限制长度
4. `namedtuple` 可读性更好的元组
5. `ChainMap` 配置优先级
6. Python 3.7+ 普通 dict 已有序


