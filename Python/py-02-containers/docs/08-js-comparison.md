# 08. Python 容器 vs JavaScript

## 📊 类型对照表

| Python | JavaScript | 说明 |
|--------|------------|------|
| `list` | `Array` | 有序可变 |
| `tuple` | 无直接对应 | 有序不可变 |
| `dict` | `Object` / `Map` | 键值对 |
| `set` | `Set` | 无序唯一 |
| `frozenset` | 无 | 不可变集合 |
| `deque` | 无 | 双端队列 |
| `Counter` | 无 | 计数器 |

---

## 📝 列表 vs 数组

### 创建

```python
# Python
lst = [1, 2, 3]
lst = list(range(5))
lst = [0] * 5
```

```javascript
// JavaScript
const arr = [1, 2, 3];
const arr = Array.from({length: 5}, (_, i) => i);
const arr = Array(5).fill(0);
```

### 操作对照

| Python | JavaScript |
|--------|------------|
| `lst.append(x)` | `arr.push(x)` |
| `lst.extend([a, b])` | `arr.push(a, b)` |
| `lst.insert(0, x)` | `arr.unshift(x)` |
| `lst.pop()` | `arr.pop()` |
| `lst.pop(0)` | `arr.shift()` |
| `lst.remove(x)` | `arr.splice(arr.indexOf(x), 1)` |
| `x in lst` | `arr.includes(x)` |
| `lst.index(x)` | `arr.indexOf(x)` |
| `lst.count(x)` | `arr.filter(i => i === x).length` |
| `len(lst)` | `arr.length` |
| `lst.sort()` | `arr.sort()` |
| `lst.reverse()` | `arr.reverse()` |
| `lst.copy()` | `[...arr]` 或 `arr.slice()` |

### 切片

```python
# Python 切片
lst[1:4]      # 索引 1-3
lst[::2]      # 每隔一个
lst[::-1]     # 反转
```

```javascript
// JavaScript
arr.slice(1, 4);
arr.filter((_, i) => i % 2 === 0);
arr.slice().reverse();  // 或 [...arr].reverse()
```

---

## 📖 字典 vs 对象/Map

### 创建

```python
# Python
d = {"name": "Alice", "age": 25}
d = dict(name="Alice", age=25)
```

```javascript
// JavaScript Object
const obj = {name: "Alice", age: 25};

// JavaScript Map
const map = new Map([["name", "Alice"], ["age", 25]]);
```

### 操作对照（vs Object）

| Python | JavaScript |
|--------|------------|
| `d["key"]` | `obj.key` 或 `obj["key"]` |
| `d.get("key", default)` | `obj.key ?? default` |
| `"key" in d` | `"key" in obj` |
| `del d["key"]` | `delete obj.key` |
| `d.keys()` | `Object.keys(obj)` |
| `d.values()` | `Object.values(obj)` |
| `d.items()` | `Object.entries(obj)` |
| `d.update(d2)` | `Object.assign(obj, obj2)` |
| `{**d1, **d2}` | `{...obj1, ...obj2}` |
| `len(d)` | `Object.keys(obj).length` |

### 遍历

```python
# Python
for k, v in d.items():
    print(k, v)
```

```javascript
// JavaScript
for (const [k, v] of Object.entries(obj)) {
    console.log(k, v);
}
```

### 何时用 Map vs Object

| 场景 | JS 推荐 | Python |
|------|--------|--------|
| 简单字符串键 | Object | dict |
| 动态键/非字符串键 | Map | dict |
| 需要保持顺序 | Map | dict (3.7+) |
| 需要 JSON 序列化 | Object | dict |

---

## 🔵 集合 vs Set

### 操作对照

| Python | JavaScript |
|--------|------------|
| `s.add(x)` | `set.add(x)` |
| `s.remove(x)` | `set.delete(x)` |
| `s.discard(x)` | `set.delete(x)` |
| `x in s` | `set.has(x)` |
| `len(s)` | `set.size` |
| `s.clear()` | `set.clear()` |

### 集合运算

```python
# Python 原生支持
a | b        # 并集
a & b        # 交集
a - b        # 差集
a ^ b        # 对称差集
```

```javascript
// JavaScript 需要手动实现
const union = new Set([...a, ...b]);
const intersection = new Set([...a].filter(x => b.has(x)));
const difference = new Set([...a].filter(x => !b.has(x)));
```

---

## 🔄 推导式 vs 数组方法

### map

```python
# Python
[x * 2 for x in lst]
```

```javascript
// JavaScript
arr.map(x => x * 2);
```

### filter

```python
# Python
[x for x in lst if x > 0]
```

```javascript
// JavaScript
arr.filter(x => x > 0);
```

### map + filter

```python
# Python
[x * 2 for x in lst if x > 0]
```

```javascript
// JavaScript
arr.filter(x => x > 0).map(x => x * 2);
```

### reduce

```python
# Python
from functools import reduce
reduce(lambda acc, x: acc + x, lst, 0)
```

```javascript
// JavaScript
arr.reduce((acc, x) => acc + x, 0);
```

---

## 🔧 常用函数对照

| Python | JavaScript |
|--------|------------|
| `len(x)` | `x.length` |
| `range(n)` | `[...Array(n).keys()]` |
| `enumerate(lst)` | `lst.entries()` |
| `zip(a, b)` | `a.map((x, i) => [x, b[i]])` |
| `sorted(lst)` | `[...arr].sort()` |
| `reversed(lst)` | `[...arr].reverse()` |
| `sum(lst)` | `arr.reduce((a, b) => a + b, 0)` |
| `min(lst)` | `Math.min(...arr)` |
| `max(lst)` | `Math.max(...arr)` |
| `any(cond for x in lst)` | `arr.some(x => cond)` |
| `all(cond for x in lst)` | `arr.every(x => cond)` |

---

## ⚠️ 关键差异

### 1. 空容器的 Truthy/Falsy

```python
# Python: 空容器是 Falsy
if []:
    print("不会执行")
if {}:
    print("不会执行")
```

```javascript
// JavaScript: 空容器是 Truthy！
if ([]) {
    console.log("会执行！");
}
if ({}) {
    console.log("会执行！");
}
```

### 2. 负索引

```python
# Python 原生支持
lst[-1]   # 最后一个
lst[-2]   # 倒数第二个
```

```javascript
// JavaScript 需要 at() (ES2022+)
arr.at(-1);
// 或
arr[arr.length - 1];
```

### 3. 不可变数据结构

```python
# Python 有不可变版本
tuple   # 不可变列表
frozenset  # 不可变集合
```

```javascript
// JavaScript 没有内置不可变结构
// 需要使用 Object.freeze() 或 Immutable.js
```

### 4. 字典/对象键的类型

```python
# Python 字典键可以是任何可哈希类型
d = {
    1: "int",
    "key": "str",
    (1, 2): "tuple",
}
```

```javascript
// JavaScript Object 键总是字符串（或 Symbol）
const obj = {1: "int"};  // 键变成字符串 "1"

// Map 可以用任何类型作为键
const map = new Map([[{}, "object key"]]);
```

---

## ✅ 快速对照记忆

| 操作 | Python | JavaScript |
|------|--------|------------|
| 添加末尾 | `append` | `push` |
| 添加开头 | `insert(0, x)` | `unshift` |
| 删除末尾 | `pop()` | `pop()` |
| 删除开头 | `pop(0)` | `shift()` |
| 切片 | `lst[1:3]` | `arr.slice(1, 3)` |
| 反转 | `lst[::-1]` | `arr.reverse()` |
| 映射 | `[f(x) for x in lst]` | `arr.map(f)` |
| 过滤 | `[x for x in lst if cond]` | `arr.filter(x => cond)` |
| 归约 | `reduce(f, lst, init)` | `arr.reduce(f, init)` |
| 解构 | `a, b = lst` | `const [a, b] = arr` |

