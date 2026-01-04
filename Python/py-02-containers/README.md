# P02: 容器与数据结构

> 面向 JS/TS 资深工程师的 Python 容器类型教程

## 🎯 学完后能做

- ✅ 熟练使用 list/tuple/dict/set
- ✅ 写出 Pythonic 的推导式代码
- ✅ 理解可变/不可变、可哈希的概念

---

## 🚀 快速开始

```bash
# 进入示例目录
cd examples

# 运行列表示例
python3 01_list_demo.py

# 运行所有示例
cd ../scripts && bash run_all.sh
```

---

## 📚 目录结构

```
py-02-containers/
├── README.md
├── docs/
│   ├── 01-list.md                # 列表
│   ├── 02-tuple.md               # 元组
│   ├── 03-dict.md                # 字典
│   ├── 04-set.md                 # 集合
│   ├── 05-comprehensions.md      # 推导式
│   ├── 06-sequence-operations.md # 序列操作
│   ├── 07-mutable-immutable.md   # 可变与不可变
│   ├── 08-js-comparison.md       # JS 对照
│   ├── 09-exercises.md           # 练习题
│   └── 10-interview-questions.md # 面试题
├── examples/
├── exercises/
├── project/
│   └── word_frequency/
└── scripts/
```

---

## ⚡ Python 容器 vs JavaScript

| Python | JavaScript | 说明 |
|--------|------------|------|
| `list` | `Array` | 有序可变序列 |
| `tuple` | 无直接对应 | 有序不可变序列 |
| `dict` | `Object` / `Map` | 键值对映射 |
| `set` | `Set` | 无序不重复集合 |
| `frozenset` | 无 | 不可变集合 |

---

## 🔥 核心概念速查

### 列表 list

```python
# 创建
lst = [1, 2, 3]
lst = list(range(5))  # [0, 1, 2, 3, 4]

# 操作
lst.append(4)         # 末尾添加
lst.extend([5, 6])    # 扩展
lst.pop()             # 弹出末尾
lst.insert(0, 0)      # 指定位置插入

# 切片
lst[1:3]              # [1, 2]
lst[::-1]             # 反转
lst[::2]              # 每隔一个

# 排序
lst.sort()            # 原地排序
sorted(lst)           # 返回新列表
```

### 元组 tuple

```python
# 创建（不可变）
t = (1, 2, 3)
t = (1,)              # 单元素必须有逗号

# 解包
a, b, c = (1, 2, 3)
first, *rest = (1, 2, 3, 4)  # first=1, rest=[2,3,4]

# 命名元组
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)
```

### 字典 dict

```python
# 创建
d = {"a": 1, "b": 2}
d = dict(a=1, b=2)

# 操作
d["c"] = 3            # 添加/修改
d.get("x", 0)         # 安全获取
d.setdefault("x", []) # 不存在则设置
d.update({"d": 4})    # 合并

# 遍历
for k, v in d.items():
    print(k, v)

# 推导式
squares = {x: x**2 for x in range(5)}
```

### 集合 set

```python
# 创建
s = {1, 2, 3}
s = set()             # 空集合（{} 是空字典！）

# 操作
s.add(4)
s.remove(1)           # 不存在会报错
s.discard(1)          # 不存在不报错

# 集合运算
a | b                 # 并集
a & b                 # 交集
a - b                 # 差集
a ^ b                 # 对称差集
```

### 推导式

```python
# 列表推导式
[x**2 for x in range(10)]
[x for x in range(10) if x % 2 == 0]

# 字典推导式
{k: v for k, v in pairs}

# 集合推导式
{x**2 for x in range(10)}

# 生成器表达式（惰性求值）
(x**2 for x in range(10))
```

---

## ⚠️ 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| **空集合** | `{}` 是空字典 | 用 `set()` |
| **浅拷贝** | `lst.copy()` 只复制一层 | 用 `copy.deepcopy()` |
| **遍历修改** | 遍历时修改列表会出问题 | 遍历副本或使用推导式 |
| **默认参数** | `def f(lst=[])` 共享 | 用 `lst=None` |
| **dict.keys()** | 返回视图不是列表 | 需要时 `list(d.keys())` |

---

## 📖 学习路径

1. [列表 list](docs/01-list.md)
2. [元组 tuple](docs/02-tuple.md)
3. [字典 dict](docs/03-dict.md)
4. [集合 set](docs/04-set.md)
5. [推导式](docs/05-comprehensions.md)
6. [序列操作](docs/06-sequence-operations.md)
7. [可变与不可变](docs/07-mutable-immutable.md)
8. [JS 对照](docs/08-js-comparison.md)
9. [练习题](docs/09-exercises.md)
10. [面试题](docs/10-interview-questions.md)

---

## 🛠️ 小项目：词频统计器

```bash
python3 project/word_frequency/main.py sample.txt --top 10
```

输出 Top 10 高频词及其出现次数。

