# 10. 面试高频问题

> 10 个 Python 容器相关的面试高频问题

---

## 1. list 和 tuple 的区别？什么时候用 tuple？

<details>
<summary>参考答案</summary>

| 特性 | list | tuple |
|------|------|-------|
| 可变性 | 可变 | 不可变 |
| 语法 | `[1, 2, 3]` | `(1, 2, 3)` |
| 可作为 dict 键 | ❌ | ✅ |
| 方法数量 | 多 | 少 |
| 内存占用 | 较大 | 较小 |
| 性能 | 略慢 | 略快 |

**使用 tuple 的场景**：
1. 数据不应被修改（如坐标、配置）
2. 需要作为字典键
3. 函数返回多个值
4. 需要解包操作

```python
# 好的 tuple 用例
point = (10, 20)
RGB_RED = (255, 0, 0)
def get_stats(): return min_val, max_val, avg_val

# 好的 list 用例
items = [1, 2, 3]  # 需要增删改
```

</details>

---

## 2. dict 的键必须满足什么条件？

<details>
<summary>参考答案</summary>

字典的键必须是**可哈希的（hashable）**。

**可哈希的条件**：
1. 有 `__hash__()` 方法
2. 有 `__eq__()` 方法
3. 生命周期内哈希值不变

**可哈希类型**：`int`, `float`, `str`, `tuple`（元素都可哈希时）, `frozenset`

**不可哈希类型**：`list`, `dict`, `set`

```python
# ✅ 可以作为键
d[1] = "int"
d["key"] = "str"
d[(1, 2)] = "tuple"

# ❌ 不能作为键
d[[1, 2]] = "list"  # TypeError
```

**为什么 list 不能作为键？**

因为 list 是可变的，如果允许 list 作为键，修改 list 后哈希值会变，导致字典内部结构混乱。

</details>

---

## 3. 如何合并两个字典？

<details>
<summary>参考答案</summary>

```python
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}

# 方式 1：update()（原地修改 d1）
d1.update(d2)

# 方式 2：解包（创建新字典）
d3 = {**d1, **d2}

# 方式 3：| 运算符（Python 3.9+）
d3 = d1 | d2

# 方式 4：|= 原地合并（Python 3.9+）
d1 |= d2
```

**注意**：相同的键，后面的值会覆盖前面的。

</details>

---

## 4. 推导式和 map/filter 哪个更 Pythonic？

<details>
<summary>参考答案</summary>

**推导式更 Pythonic**。

```python
# map 方式
squares = list(map(lambda x: x**2, range(10)))

# 推导式（更清晰）
squares = [x**2 for x in range(10)]

# filter + map
result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))

# 推导式（更简洁）
result = [x**2 for x in range(10) if x % 2 == 0]
```

**但以下情况可以用 map/filter**：
- 已有现成函数时：`list(map(str, numbers))`
- 需要惰性求值时（不转 list）

</details>

---

## 5. 浅拷贝和深拷贝的区别？

<details>
<summary>参考答案</summary>

**浅拷贝**：只复制一层，嵌套对象仍是引用。

```python
original = [1, 2, [3, 4]]
shallow = original.copy()  # 或 original[:]

shallow[2][0] = 100
print(original)  # [1, 2, [100, 4]] ← 被修改了！
```

**深拷贝**：递归复制所有层级。

```python
import copy
deep = copy.deepcopy(original)

deep[2][0] = 100
print(original)  # [1, 2, [3, 4]] ← 不受影响
```

**何时用深拷贝**：
- 嵌套的列表/字典
- 自定义对象
- 需要完全独立的副本

</details>

---

## 6. 如何去重并保持顺序？

<details>
<summary>参考答案</summary>

```python
lst = [1, 2, 2, 3, 1, 4, 3]

# 方式 1：dict.fromkeys（Python 3.7+）
unique = list(dict.fromkeys(lst))

# 方式 2：手动去重
seen = set()
unique = [x for x in lst if not (x in seen or seen.add(x))]

# 方式 3：传统循环
seen = set()
unique = []
for x in lst:
    if x not in seen:
        seen.add(x)
        unique.append(x)

print(unique)  # [1, 2, 3, 4]
```

**注意**：`set(lst)` 会去重但不保序。

</details>

---

## 7. 如何找出两个列表的交集？

<details>
<summary>参考答案</summary>

```python
a = [1, 2, 3, 4, 5]
b = [4, 5, 6, 7, 8]

# 方式 1：集合运算（推荐）
intersection = list(set(a) & set(b))

# 方式 2：推导式
intersection = [x for x in a if x in set(b)]

# 方式 3：保持 a 的顺序
set_b = set(b)
intersection = [x for x in a if x in set_b]

print(intersection)  # [4, 5]
```

</details>

---

## 8. dict.get() 和直接 [] 访问的区别？

<details>
<summary>参考答案</summary>

```python
d = {"name": "Alice"}

# [] 访问：不存在会报错
d["name"]    # "Alice"
d["age"]     # ❌ KeyError

# get() 访问：不存在返回 None 或默认值
d.get("name")       # "Alice"
d.get("age")        # None
d.get("age", 0)     # 0

# setdefault()：不存在则设置并返回
d.setdefault("age", 25)  # 25（同时设置了 d["age"] = 25）
```

**使用建议**：
- 确定键存在：用 `[]`
- 可能不存在：用 `get()`
- 不存在则初始化：用 `setdefault()` 或 `defaultdict`

</details>

---

## 9. Python 的 dict 是如何实现的？

<details>
<summary>参考答案</summary>

Python 的 dict 使用**哈希表**实现。

**工作原理**：
1. 计算键的哈希值
2. 用哈希值找到存储位置（索引）
3. 处理哈希冲突（开放寻址法）

**时间复杂度**：
- 查找：O(1) 平均，O(n) 最坏
- 插入：O(1) 平均
- 删除：O(1) 平均

**Python 3.7+ 的改进**：
- 字典保持插入顺序
- 内存更紧凑

**为什么 dict 查找快？**
```python
# O(1) - 直接哈希定位
value = d["key"]

# O(n) - 需要遍历
value = None
for item in lst:
    if item["key"] == "target":
        value = item
```

</details>

---

## 10. 为什么字符串可以作为 dict 的键，列表不行？

<details>
<summary>参考答案</summary>

**字符串是不可变的**：
- 创建后内容不能修改
- 哈希值固定不变
- 可以安全地作为字典键

**列表是可变的**：
- 内容可以随时修改
- 如果允许作为键，修改后哈希值会变
- 导致字典无法正确定位数据

```python
# 假设允许 list 作为键（实际不允许）
d = {}
key = [1, 2]
d[key] = "value"  # 假设哈希值是 100

key.append(3)     # 修改后哈希值变成 200
d[key]            # 找不到！因为位置变了
```

**总结**：可变对象的哈希值不稳定，所以不能作为字典键。

**替代方案**：将 list 转为 tuple
```python
d = {}
d[tuple([1, 2, 3])] = "value"  # ✅
```

</details>

