# 06. 练习题

## 练习 1：使用 pdb 调试

调试以下代码，找出 bug：

```python
def calculate_average(numbers):
    total = 0
    count = 0
    for n in numbers:
        total += n
        count += 1
    return total / count

# 调用时出错
result = calculate_average([])
```

使用 breakpoint() 或 pdb 找出问题。

---

## 练习 2：配置日志

为以下应用配置日志：

1. 控制台输出 INFO 及以上
2. 文件输出 DEBUG 及以上
3. 文件轮转，最大 10MB，保留 5 个

```python
import logging

def setup_logging():
    # 完成配置
    pass
```

---

## 练习 3：分析性能

使用 cProfile 分析以下代码的性能瓶颈：

```python
def slow_function():
    result = []
    for i in range(10000):
        for j in range(100):
            result.append(i * j)
    return result

def process():
    data = slow_function()
    return sum(data)
```

---

## 练习 4：内存分析

使用 tracemalloc 找出以下代码的内存使用情况：

```python
def memory_test():
    data = []
    for i in range(100):
        data.append([0] * 10000)
    return data
```

---

## 练习 5：优化字符串拼接

优化以下代码：

```python
def build_html(items):
    html = "<ul>"
    for item in items:
        html += f"<li>{item}</li>"
    html += "</ul>"
    return html
```

---

## 练习 6：优化查找

优化以下代码：

```python
def find_duplicates(items):
    duplicates = []
    for i, item in enumerate(items):
        if item in items[i+1:]:  # O(n²)
            if item not in duplicates:
                duplicates.append(item)
    return duplicates
```

---

## 练习 7：使用 lru_cache

使用 lru_cache 优化递归斐波那契：

```python
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# fibonacci(35) 很慢
```

---

## 练习 8：定位内存泄漏

找出以下代码的内存泄漏：

```python
class EventHandler:
    handlers = []  # 类变量

    def register(self, callback):
        self.handlers.append(callback)

    def process(self, event):
        for handler in self.handlers:
            handler(event)

# 每次创建新实例都注册 handler
for i in range(1000):
    handler = EventHandler()
    handler.register(lambda e: print(e))
```

---

## 练习 9：VS Code 调试配置

为以下项目创建 VS Code 调试配置：

1. 调试当前文件
2. 调试 pytest 测试
3. 调试 FastAPI 应用

---

## 练习 10：生产日志配置

创建一个生产环境的日志配置：

1. 从环境变量读取日志级别
2. 使用 JSON 格式
3. 包含时间戳、级别、消息
4. 输出到 stdout

---

## 练习 11：比较数据结构性能

比较以下场景的性能：

1. 在 list vs set 中查找元素
2. 使用 list.pop(0) vs deque.popleft()
3. dict.get() vs try/except KeyError

---

## 练习 12：优化循环

优化以下代码：

```python
import math

def process_data(data):
    results = []
    for item in data:
        if len(data) > 0:
            value = math.sqrt(item) * math.pi
            results.append(value)
    return results
```

