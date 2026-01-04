# 08. 面试题

## 1. 什么是 GIL？它对多线程有什么影响？

**答案**：

GIL (Global Interpreter Lock) 是 CPython 解释器的全局锁，确保同一时刻只有一个线程执行 Python 字节码。

**影响**：
- **CPU 密集型**: 多线程无法利用多核，甚至可能更慢
- **I/O 密集型**: 等待 I/O 时会释放 GIL，多线程仍有效

**解决方案**：
- CPU 密集：使用 `multiprocessing`
- I/O 密集：使用 `asyncio` 或 `threading`
- 使用 C 扩展释放 GIL（如 NumPy）
- 使用 PyPy（仍有 GIL 但更快）

```python
# CPU 密集型应该用多进程
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    results = executor.map(cpu_bound_func, data)
```

---

## 2. .pyc 文件是什么？什么时候重新生成？

**答案**：

.pyc 是编译后的 Python 字节码缓存文件，存储在 `__pycache__` 目录中。

**重新生成条件**：
1. 源文件 (.py) 修改时间变化
2. Python 版本变化（Magic Number 不匹配）
3. .pyc 文件不存在

```bash
# 查看 .pyc 文件
ls __pycache__/
# module.cpython-312.pyc

# 强制重新编译
python3 -m compileall -f .

# 禁止生成 .pyc
PYTHONDONTWRITEBYTECODE=1 python3 script.py
```

---

## 3. Python 的内存管理机制是什么？

**答案**：

Python 使用**引用计数 + 分代垃圾回收**：

**引用计数**：
- 每个对象有引用计数器
- 引用增加时 +1，减少时 -1
- 计数为 0 时立即释放

**分代 GC**：
- 处理循环引用（引用计数无法处理）
- 分为 3 代：0（新对象）、1、2（老对象）
- 定期检查循环引用并释放

```python
import sys
import gc

a = [1, 2, 3]
print(sys.getrefcount(a))  # 查看引用计数

gc.collect()  # 手动触发 GC
print(gc.get_count())  # 查看各代对象数量
```

---

## 4. 如何定位 Python 程序的内存泄漏？

**答案**：

**方法一：tracemalloc**
```python
import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

# 可疑代码
run_suspect_code()

gc.collect()
snapshot2 = tracemalloc.take_snapshot()

# 比较差异
diff = snapshot2.compare_to(snapshot1, 'lineno')
for stat in diff[:10]:
    print(stat)
```

**方法二：objgraph**
```python
import objgraph
objgraph.show_growth()  # 查看增长最快的类型
objgraph.show_backrefs(obj)  # 查看引用链
```

**常见泄漏原因**：
- 循环引用 + `__del__`
- 全局变量累积
- 闭包捕获大对象
- 缓存无限增长

---

## 5. async 的本质是什么？

**答案**：

async 是**协作式多任务**，基于协程实现。

**本质**：
- `async def` 定义协程函数，返回协程对象
- `await` 暂停当前协程，让出控制权
- 事件循环调度多个协程

**与线程区别**：
- 线程：抢占式，OS 调度
- async：协作式，程序员控制

```python
import asyncio

async def task():
    await asyncio.sleep(1)  # 让出控制权
    return "done"

# 本质：协程对象
coro = task()
print(type(coro))  # <class 'coroutine'>

# 需要事件循环执行
result = asyncio.run(task())
```

---

## 6. 什么时候用多进程？什么时候用多线程？

**答案**：

| 场景 | 推荐 | 原因 |
|------|------|------|
| CPU 密集 | multiprocessing | 绕过 GIL |
| I/O 密集（少量） | threading | 简单 |
| I/O 密集（高并发） | asyncio | 高效 |
| 混合场景 | 进程池 + asyncio | 最佳组合 |

**示例**：
```python
# CPU 密集
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    results = executor.map(compute, data)

# I/O 密集
import asyncio
async def main():
    tasks = [fetch(url) for url in urls]
    await asyncio.gather(*tasks)
```

---

## 7. import 时 Python 如何查找模块？

**答案**：

按 `sys.path` 列表顺序查找：

1. **当前脚本目录**
2. **PYTHONPATH 环境变量**
3. **标准库目录**
4. **site-packages**（第三方包）

```python
import sys
for path in sys.path:
    print(path)
```

**常见问题**：
- 同名模块冲突：当前目录优先
- 相对导入失败：用 `python -m` 运行
- 包不在路径：修改 PYTHONPATH 或 sys.path

---

## 8. `__name__ == "__main__"` 的作用？

**答案**：

判断模块是**直接运行**还是**被导入**。

```python
# mymodule.py
def main():
    print("Main function")

if __name__ == "__main__":
    # 只有直接运行时执行
    main()
```

**运行方式**：
```bash
python mymodule.py     # __name__ = "__main__"，执行 main()
python -c "import mymodule"  # __name__ = "mymodule"，不执行
```

**用途**：
- 模块测试代码
- 命令行入口
- 避免导入时的副作用

---

## 9. 如何优化 Python 程序性能？

**答案**：

**代码层面**：
1. 使用局部变量（比全局快）
2. 使用列表推导式（比循环快）
3. 使用 `''.join()` 拼接字符串
4. 使用生成器（节省内存）
5. 缓存计算结果（`@lru_cache`）

**工具层面**：
1. 使用 NumPy（C 扩展）
2. 使用 PyPy（JIT 编译）
3. 使用 Cython（编译为 C）
4. 使用 Numba（JIT for NumPy）

**架构层面**：
1. 多进程处理 CPU 密集
2. 异步处理 I/O 密集
3. 使用缓存（Redis）

```python
# 优化示例
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

---

## 10. PyPy 和 CPython 的区别？

**答案**：

| 特性 | CPython | PyPy |
|------|---------|------|
| 实现语言 | C | Python (RPython) |
| 执行方式 | 解释执行 | JIT 编译 |
| 运行速度 | 慢 | 快 (3-10x) |
| 启动速度 | 快 | 慢 |
| 内存使用 | 少 | 多 |
| C 扩展 | 完美支持 | 有限支持 |
| 适用场景 | 通用 | 长时间运行 |

**选择建议**：
- 需要 C 扩展 → CPython
- 长时间计算 → PyPy
- 快速脚本 → CPython
- 内存受限 → CPython

---

## 11. 练习题（15 道）

### 字节码相关

1. 使用 `dis` 模块反汇编一个简单函数，解释每条指令的含义。
2. 编写代码比较 `a + b` 和 `a.__add__(b)` 的字节码差异。
3. 使用 `compile()` 和 `exec()` 动态执行代码。

### GIL 与并发

4. 编写程序证明 GIL 对 CPU 密集型任务的影响。
5. 使用 `threading` 实现并发下载。
6. 使用 `multiprocessing` 实现并行计算。
7. 使用 `asyncio` 实现高并发 HTTP 请求。
8. 比较同一任务在 threading、multiprocessing、asyncio 下的性能。

### import 系统

9. 创建一个包含子包的模块，练习相对导入。
10. 使用 `importlib` 动态导入模块。
11. 实现一个简单的插件加载系统。

### 内存与性能

12. 使用 `tracemalloc` 追踪代码的内存使用。
13. 使用 `cProfile` 分析函数性能瓶颈。
14. 找出并修复一个循环引用导致的内存泄漏。
15. 使用 `__slots__` 优化类的内存使用。

