# 08. Python vs JavaScript 对照表

> 面向 JS/TS 开发者的快速参考

---

## 📝 基础语法

| 特性 | Python | JavaScript |
|------|--------|------------|
| 代码块 | 缩进 | `{ }` |
| 语句结束 | 无需分号 | `;`（可选） |
| 注释 | `# 单行` / `"""多行"""` | `// 单行` / `/* 多行 */` |
| 常量 | `UPPER_CASE`（约定） | `const` |
| 空值 | `None` | `null` / `undefined` |
| 布尔值 | `True` / `False` | `true` / `false` |

---

## 🔢 变量与类型

```python
# Python
name = "Alice"        # 无需 let/const
age = 25
is_valid = True
data = None
```

```javascript
// JavaScript
let name = "Alice";
const age = 25;
let isValid = true;
let data = null;
```

### 类型检查

| Python | JavaScript |
|--------|------------|
| `type(x)` | `typeof x` |
| `isinstance(x, int)` | `typeof x === 'number'` |
| `isinstance(x, list)` | `Array.isArray(x)` |

---

## ➕ 运算符

| 操作 | Python | JavaScript |
|------|--------|------------|
| 整除 | `//` | `Math.floor(a/b)` |
| 幂运算 | `**` | `**` |
| 取模 | `%` | `%` |
| 逻辑与 | `and` | `&&` |
| 逻辑或 | `or` | `\|\|` |
| 逻辑非 | `not` | `!` |
| 相等 | `==` | `===` |
| 不等 | `!=` | `!==` |
| 身份 | `is` | 无 |
| 成员 | `in` | `includes()` / `in` |
| 三元 | `a if cond else b` | `cond ? a : b` |

---

## 📝 字符串

```python
# Python
name = "Alice"
greeting = f"Hello, {name}!"         # f-string
multi = """多
行"""
upper = name.upper()
items = "a,b,c".split(",")           # ['a', 'b', 'c']
joined = ",".join(["a", "b", "c"])   # "a,b,c"
trimmed = "  hi  ".strip()           # "hi"
```

```javascript
// JavaScript
const name = "Alice";
const greeting = `Hello, ${name}!`;  // 模板字符串
const multi = `多
行`;
const upper = name.toUpperCase();
const items = "a,b,c".split(",");    // ['a', 'b', 'c']
const joined = ["a", "b", "c"].join(","); // "a,b,c"
const trimmed = "  hi  ".trim();     // "hi"
```

### 字符串方法对照

| Python | JavaScript |
|--------|------------|
| `s.upper()` | `s.toUpperCase()` |
| `s.lower()` | `s.toLowerCase()` |
| `s.strip()` | `s.trim()` |
| `s.split(",")` | `s.split(",")` |
| `",".join(arr)` | `arr.join(",")` |
| `s.replace(a, b)` | `s.replace(a, b)` |
| `s.find(x)` | `s.indexOf(x)` |
| `s.startswith(x)` | `s.startsWith(x)` |
| `s.endswith(x)` | `s.endsWith(x)` |
| `len(s)` | `s.length` |
| `s[0]` | `s[0]` 或 `s.charAt(0)` |
| `s[-1]` | `s.at(-1)` |
| `s[1:4]` | `s.slice(1, 4)` |
| `s[::-1]` | `s.split('').reverse().join('')` |

---

## 🔄 控制流

### 条件语句

```python
# Python
if age < 18:
    print("未成年")
elif age < 60:
    print("成年")
else:
    print("老年")
```

```javascript
// JavaScript
if (age < 18) {
    console.log("未成年");
} else if (age < 60) {
    console.log("成年");
} else {
    console.log("老年");
}
```

### for 循环

```python
# Python
for item in items:           # 遍历
    print(item)

for i in range(5):           # 0-4
    print(i)

for i, item in enumerate(items):  # 带索引
    print(i, item)
```

```javascript
// JavaScript
for (const item of items) {  // 遍历
    console.log(item);
}

for (let i = 0; i < 5; i++) { // 0-4
    console.log(i);
}

items.forEach((item, i) => { // 带索引
    console.log(i, item);
});
```

### while 循环

```python
# Python
while condition:
    do_something()
```

```javascript
// JavaScript
while (condition) {
    doSomething();
}
```

---

## 🔧 函数

### 定义

```python
# Python
def add(a, b):
    return a + b

# 默认参数
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Lambda
square = lambda x: x ** 2
```

```javascript
// JavaScript
function add(a, b) {
    return a + b;
}

// 默认参数
function greet(name, greeting = "Hello") {
    return `${greeting}, ${name}!`;
}

// 箭头函数
const square = x => x ** 2;
```

### 可变参数

```python
# Python
def sum_all(*args):
    return sum(args)

def print_info(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}: {v}")
```

```javascript
// JavaScript
function sumAll(...args) {
    return args.reduce((a, b) => a + b, 0);
}

function printInfo(obj) {
    for (const [k, v] of Object.entries(obj)) {
        console.log(`${k}: ${v}`);
    }
}
```

---

## 📦 数据结构

| Python | JavaScript |
|--------|------------|
| `list` | `Array` |
| `dict` | `Object` / `Map` |
| `set` | `Set` |
| `tuple` | 无（用数组） |

### 列表/数组

```python
# Python
arr = [1, 2, 3]
arr.append(4)           # 末尾添加
arr.pop()               # 弹出末尾
arr.insert(0, 0)        # 指定位置插入
del arr[0]              # 删除
len(arr)                # 长度
arr[0]                  # 索引
arr[-1]                 # 最后一个
arr[1:3]                # 切片
```

```javascript
// JavaScript
const arr = [1, 2, 3];
arr.push(4);            // 末尾添加
arr.pop();              // 弹出末尾
arr.unshift(0);         // 开头添加
arr.splice(0, 1);       // 删除
arr.length;             // 长度
arr[0];                 // 索引
arr.at(-1);             // 最后一个
arr.slice(1, 3);        // 切片
```

### 字典/对象

```python
# Python
d = {"name": "Alice", "age": 25}
d["name"]               # 访问
d["city"] = "NYC"       # 添加
del d["city"]           # 删除
"name" in d             # 检查键
d.keys()                # 所有键
d.values()              # 所有值
d.items()               # 键值对
d.get("name", "N/A")    # 带默认值获取
```

```javascript
// JavaScript
const d = {name: "Alice", age: 25};
d.name;                 // 访问
d.city = "NYC";         // 添加
delete d.city;          // 删除
"name" in d;            // 检查键
Object.keys(d);         // 所有键
Object.values(d);       // 所有值
Object.entries(d);      // 键值对
d.name ?? "N/A";        // 带默认值获取
```

---

## 🎭 Truthy / Falsy

| Python Falsy | JavaScript Falsy |
|--------------|------------------|
| `False` | `false` |
| `None` | `null`, `undefined` |
| `0`, `0.0` | `0`, `-0`, `0n` |
| `""` | `""` |
| `[]`, `{}`, `set()` | **Truthy!** |
| — | `NaN` |

**⚠️ 重要差异**：Python 空容器是 Falsy，JS 空数组/对象是 Truthy！

---

## 📁 模块导入

```python
# Python
import math
from math import sqrt
from math import sqrt as s
from math import *

import json
data = json.loads('{}')
```

```javascript
// JavaScript (ESM)
import math from 'math';
import { sqrt } from 'math';
import { sqrt as s } from 'math';
import * as math from 'math';

import fs from 'fs';

// CommonJS
const math = require('math');
const { sqrt } = require('math');
```

---

## 🎯 常见模式对照

### 数组操作

| 操作 | Python | JavaScript |
|------|--------|------------|
| 映射 | `[x*2 for x in arr]` | `arr.map(x => x*2)` |
| 过滤 | `[x for x in arr if x > 0]` | `arr.filter(x => x > 0)` |
| 归约 | `sum(arr)` | `arr.reduce((a,b) => a+b, 0)` |
| 查找 | `next((x for x in arr if x > 0), None)` | `arr.find(x => x > 0)` |
| 所有 | `all(x > 0 for x in arr)` | `arr.every(x => x > 0)` |
| 任一 | `any(x > 0 for x in arr)` | `arr.some(x => x > 0)` |

### 解构

```python
# Python
a, b = [1, 2]
a, *rest = [1, 2, 3, 4]  # a=1, rest=[2,3,4]
x, y = y, x              # 交换
```

```javascript
// JavaScript
const [a, b] = [1, 2];
const [a, ...rest] = [1, 2, 3, 4]; // a=1, rest=[2,3,4]
[x, y] = [y, x];         // 交换
```

---

## ✅ 快速记忆

1. **缩进** 代替 `{ }`
2. **True/False** 首字母大写
3. **and/or/not** 代替 `&&/||/!`
4. **elif** 代替 `else if`
5. **None** 代替 `null`
6. **range(n)** 代替 `for (let i=0; i<n; i++)`
7. **f"..."** 代替 `` `...` ``
8. **def** 代替 `function`
9. **空容器是 Falsy**

