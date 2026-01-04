# 05. 内存与 GC

## 本节目标

- 理解 Python 的内存管理机制
- 掌握引用计数和垃圾回收
- 学会诊断内存问题

---

## 引用计数

Python 主要使用**引用计数**管理内存。

### 基本原理

```python
import sys

a = [1, 2, 3]
print(sys.getrefcount(a))  # 2（a 本身 + getrefcount 参数）

b = a  # 增加引用
print(sys.getrefcount(a))  # 3

del b  # 减少引用
print(sys.getrefcount(a))  # 2

# 引用计数为 0 时，对象被立即释放
```

### 引用计数变化

```python
import sys

def show_refcount(obj):
    # 注意：调用函数本身会增加引用
    print(f"refcount: {sys.getrefcount(obj)}")

x = [1, 2, 3]

# 增加引用的操作
y = x           # 赋值
lst = [x]       # 放入容器
func(x)         # 传递给函数

# 减少引用的操作
del y           # 删除变量
lst.pop()       # 从容器移除
# 函数返回后     # 局部变量销毁
```

---

## 对象的 id 和 is

### id() 函数

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(id(a))  # 对象的内存地址
print(id(b))  # 不同对象，不同地址
print(id(c))  # 同一对象，相同地址

print(a is b)  # False
print(a is c)  # True
```

### 小整数缓存

```python
# CPython 缓存 -5 到 256 的整数
a = 100
b = 100
print(a is b)  # True（同一对象）

a = 1000
b = 1000
print(a is b)  # False（不同对象，可能）
```

### 字符串驻留

```python
# 短字符串会被驻留
a = "hello"
b = "hello"
print(a is b)  # True

a = "hello world!"
b = "hello world!"
print(a is b)  # False（含特殊字符）
```

---

## 循环引用问题

引用计数无法处理循环引用：

```python
class Node:
    def __init__(self):
        self.ref = None

    def __del__(self):
        print("Node deleted")

# 创建循环引用
a = Node()
b = Node()
a.ref = b
b.ref = a

# 删除变量
del a
del b
# Node 对象不会被释放！因为引用计数不为 0
```

---

## 垃圾回收 (GC)

Python 使用**分代垃圾回收**处理循环引用。

### gc 模块

```python
import gc

# 手动触发 GC
gc.collect()

# 查看当前对象数量
print(gc.get_count())  # (gen0, gen1, gen2)

# 获取阈值
print(gc.get_threshold())  # 默认 (700, 10, 10)

# 设置阈值
gc.set_threshold(1000, 15, 15)

# 禁用/启用 GC
gc.disable()
gc.enable()

# 检查是否启用
print(gc.isenabled())
```

### 分代回收

```
Generation 0: 新创建的对象
    ↓ 存活后晋升
Generation 1: 存活过一次的对象
    ↓ 存活后晋升
Generation 2: 长期存活的对象
```

```python
import gc

# 查看各代对象数量
print(gc.get_count())  # (387, 3, 0)

# 收集指定代
gc.collect(0)  # 只收集第 0 代
gc.collect(1)  # 收集第 0 和第 1 代
gc.collect(2)  # 收集所有代（完整 GC）
```

### 检测循环引用

```python
import gc

# 设置调试标志
gc.set_debug(gc.DEBUG_LEAK)

class Node:
    def __init__(self):
        self.ref = None

a = Node()
b = Node()
a.ref = b
b.ref = a

del a, b

# 手动收集
gc.collect()

# 查看不可回收的对象
print(gc.garbage)
```

---

## 弱引用

使用弱引用避免循环引用：

```python
import weakref

class Node:
    def __init__(self, name):
        self.name = name
        self.ref = None

    def __del__(self):
        print(f"Node {self.name} deleted")

# 使用弱引用
a = Node("A")
b = Node("B")

# 强引用
a.strong_ref = b

# 弱引用（不增加引用计数）
b.weak_ref = weakref.ref(a)

# 访问弱引用
ref = b.weak_ref()  # 返回对象或 None
if ref is not None:
    print(ref.name)
```

### WeakValueDictionary

```python
import weakref

class Cache:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value):
        self._cache[key] = value

# 当 value 没有其他引用时，自动从缓存移除
```

---

## tracemalloc - 内存追踪

```python
import tracemalloc

# 开始追踪
tracemalloc.start()

# 运行代码
data = [i ** 2 for i in range(100000)]

# 获取当前快照
snapshot = tracemalloc.take_snapshot()

# 按文件统计
top_stats = snapshot.statistics('lineno')
print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)

# 获取当前内存使用
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")

# 停止追踪
tracemalloc.stop()
```

### 比较快照

```python
import tracemalloc

tracemalloc.start()

snapshot1 = tracemalloc.take_snapshot()

# 运行一些代码
data = [i for i in range(100000)]

snapshot2 = tracemalloc.take_snapshot()

# 比较差异
diff = snapshot2.compare_to(snapshot1, 'lineno')
for stat in diff[:5]:
    print(stat)
```

---

## 内存泄漏检测

### 常见泄漏原因

1. **循环引用 + `__del__`**
2. **全局变量累积**
3. **闭包捕获**
4. **缓存无限增长**
5. **未关闭的资源**

### 检测方法

```python
import gc
import tracemalloc

def detect_leaks():
    tracemalloc.start()

    # 强制 GC
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()

    # 运行可能泄漏的代码
    run_suspected_code()

    # 再次 GC
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()

    # 比较
    diff = snapshot2.compare_to(snapshot1, 'lineno')

    print("内存增长最多的位置：")
    for stat in diff[:10]:
        if stat.size_diff > 0:
            print(stat)
```

### 使用 objgraph（第三方库）

```python
# pip install objgraph
import objgraph

# 查看最常见的对象类型
objgraph.show_most_common_types(limit=10)

# 增长最快的类型
objgraph.show_growth()

# 查找特定类型的引用
objgraph.show_backrefs(obj, max_depth=3)
```

---

## 内存优化技巧

### 使用 `__slots__`

```python
class PointWithSlots:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

class PointWithoutSlots:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 使用 __slots__ 节省约 40% 内存
import sys
print(sys.getsizeof(PointWithSlots(1, 2)))     # 更小
print(sys.getsizeof(PointWithoutSlots(1, 2)))  # 更大
```

### 生成器代替列表

```python
# 列表：立即占用内存
data = [x ** 2 for x in range(1000000)]

# 生成器：按需生成
data = (x ** 2 for x in range(1000000))
```

### 使用 array 代替 list

```python
import array

# list 存储 Python 对象
lst = [1, 2, 3, 4, 5]

# array 存储原始数据
arr = array.array('i', [1, 2, 3, 4, 5])

# array 更节省内存
```

---

## 本节要点

1. **引用计数**: 主要内存管理机制，为 0 立即释放
2. **id() 和 is**: 对象身份判断
3. **循环引用**: 引用计数无法处理，需要 GC
4. **分代 GC**: 0/1/2 三代，新对象先在 0 代
5. **弱引用**: 不增加引用计数，避免循环引用
6. **tracemalloc**: 内存追踪和泄漏检测
7. **优化**: `__slots__`、生成器、array

