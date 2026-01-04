# 03. 运算符

## 🎯 本节目标

- 掌握 Python 各类运算符
- 理解 `is` 和 `==` 的区别
- 对比 JS 运算符差异

---

## ➕ 算术运算符

```python
# 基本运算
a + b   # 加法
a - b   # 减法
a * b   # 乘法
a / b   # 除法（总是返回 float）
a // b  # 整除（向下取整）
a % b   # 取模
a ** b  # 幂运算

# 示例
10 / 3    # 3.3333...（float）
10 // 3   # 3（int）
10 % 3    # 1
2 ** 10   # 1024
-7 // 2   # -4（向下取整，不是向零取整）
```

### JS 对照

| Python | JavaScript | 说明 |
|--------|------------|------|
| `/` | `/` | Python 总返回 float |
| `//` | `Math.floor(a/b)` | 整除 |
| `**` | `**` | 幂运算（ES2016+）|
| `%` | `%` | 取模（负数行为不同）|

```python
# ⚠️ 负数取模差异
-7 % 3   # Python: 2（结果符号与除数相同）
# JS: -7 % 3 = -1（结果符号与被除数相同）
```

---

## 🔍 比较运算符

```python
a == b   # 等于（值比较）
a != b   # 不等于
a < b    # 小于
a > b    # 大于
a <= b   # 小于等于
a >= b   # 大于等于
a is b   # 身份比较（同一对象）
a is not b  # 非同一对象
```

### `==` vs `is`

```python
# == 比较值
a = [1, 2, 3]
b = [1, 2, 3]
a == b  # True（值相同）
a is b  # False（不是同一个对象）

# is 比较身份（内存地址）
c = a
a is c  # True（同一个对象）

# ⚠️ 小整数缓存
x = 256
y = 256
x is y  # True（Python 缓存 -5 到 256）

x = 257
y = 257
x is y  # 可能 False（不在缓存范围）
```

### JS 对照

| Python | JavaScript | 说明 |
|--------|------------|------|
| `==` | `===` | 值比较（Python 无类型转换）|
| `is` | 无直接对应 | 对象引用比较 |
| `!=` | `!==` | 不等于 |

> Python 的 `==` 类似 JS 的 `===`，不做类型转换

---

## 🔗 逻辑运算符

```python
a and b  # 与（短路求值）
a or b   # 或（短路求值）
not a    # 非
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `and` | `&&` |
| `or` | `\|\|` |
| `not` | `!` |

### 短路求值

```python
# and：第一个 Falsy 就返回
0 and "hello"    # 0
"hi" and "hello" # "hello"

# or：第一个 Truthy 就返回
0 or "hello"    # "hello"
"hi" or "hello" # "hi"

# 实际应用
name = user_name or "Anonymous"
```

---

## 🎯 成员运算符

```python
# in / not in
"a" in "abc"        # True
2 in [1, 2, 3]      # True
"key" in {"key": 1} # True（检查键）

"d" not in "abc"    # True
```

### JS 对照

| Python | JavaScript | 说明 |
|--------|------------|------|
| `x in list` | `list.includes(x)` | 数组包含 |
| `x in str` | `str.includes(x)` | 字符串包含 |
| `key in dict` | `key in obj` | 对象/字典键 |

---

## ❓ 三元运算符

### Python 语法

```python
result = value_if_true if condition else value_if_false

# 示例
status = "adult" if age >= 18 else "minor"
```

### JS 对照

```javascript
// JS 语法
const result = condition ? valueIfTrue : valueIfFalse;

const status = age >= 18 ? "adult" : "minor";
```

### 顺序对比

| Python | JavaScript |
|--------|------------|
| `真值 if 条件 else 假值` | `条件 ? 真值 : 假值` |

> Python 的顺序是"真值在前"，更接近自然语言

---

## 🔢 位运算符

```python
a & b   # 按位与
a | b   # 按位或
a ^ b   # 按位异或
~a      # 按位取反
a << n  # 左移 n 位
a >> n  # 右移 n 位
```

> 与 JS 语法相同

---

## 📊 运算符优先级

从高到低：

1. `**`（幂）
2. `~`, `+x`, `-x`（一元运算符）
3. `*`, `/`, `//`, `%`
4. `+`, `-`
5. `<<`, `>>`
6. `&`
7. `^`
8. `|`
9. `==`, `!=`, `<`, `>`, `<=`, `>=`, `is`, `in`
10. `not`
11. `and`
12. `or`

```python
# 示例
2 ** 3 ** 2   # 512（右结合：2 ** 9）
1 + 2 * 3     # 7（乘法优先）
not True and False  # False（not 优先于 and）
```

---

## 🔗 链式比较

Python 特有的语法糖：

```python
# Python 链式比较
1 < x < 10         # 等价于 1 < x and x < 10
a == b == c        # 三个都相等
x < y <= z < w     # 可以连续链接
```

```javascript
// JS 没有链式比较，需要拆开
1 < x && x < 10
```

---

## 📝 赋值运算符

```python
x = 10      # 赋值
x += 5      # x = x + 5
x -= 5      # x = x - 5
x *= 2      # x = x * 2
x /= 2      # x = x / 2
x //= 2     # x = x // 2
x %= 3      # x = x % 3
x **= 2     # x = x ** 2

# ⚠️ Python 没有 ++ 和 --
# x++  ❌ SyntaxError
x += 1  # ✅ 正确做法
```

### 海象运算符 `:=`（Python 3.8+）

```python
# 赋值表达式：在表达式中赋值
if (n := len(data)) > 10:
    print(f"数据太长：{n}")

# 列表推导中使用
results = [y for x in data if (y := expensive_func(x)) > 0]
```

---

## ✅ 本节要点

1. `/` 总是返回 float，`//` 是整除
2. `is` 比较身份（引用），`==` 比较值
3. 逻辑运算用 `and`, `or`, `not`
4. 成员检查用 `in`, `not in`
5. 三元表达式：`真值 if 条件 else 假值`
6. Python 支持链式比较
7. 没有 `++` 和 `--`

