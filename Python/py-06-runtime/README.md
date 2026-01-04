# P06: Python 运行时原理

> 面向 JS/TS 资深工程师的 Python 运行时深度解析

## 学完后能做

- 理解 Python 代码如何执行
- 正确选择 async/线程/进程
- 诊断性能和内存问题

## 快速开始

```bash
cd examples
python3 01_bytecode_demo.py
```

## 目录结构

```
py-06-runtime/
├── README.md
├── docs/
│   ├── 01-cpython-execution.md    # CPython 执行链路
│   ├── 02-bytecode-pyc.md         # 字节码与 .pyc
│   ├── 03-gil-concurrency.md      # GIL 与并发
│   ├── 04-import-system.md        # import 系统
│   ├── 05-memory-gc.md            # 内存与 GC
│   ├── 06-traceback.md            # 错误与 traceback
│   ├── 07-performance-tools.md    # 性能工具
│   └── 08-interview-questions.md  # 面试题
├── examples/
└── scripts/
```

## Python vs JavaScript 运行时对比

| 特性 | Python (CPython) | JavaScript (V8) |
|------|------------------|-----------------|
| 执行方式 | 字节码解释执行 | JIT 编译 |
| 并发模型 | GIL + 多进程 | 单线程 + 事件循环 |
| 内存管理 | 引用计数 + GC | 标记清除 GC |
| 异步 | asyncio | Promise/async-await |
| 热更新 | 可 importlib.reload | 不原生支持 |

## 核心概念速查

### CPython 执行流程

```
源代码 (.py)
    ↓ 词法/语法分析
AST (抽象语法树)
    ↓ 编译
字节码 (.pyc)
    ↓ 解释执行
Python 虚拟机
```

### GIL 选择指南

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| I/O 密集 | asyncio | 单线程高效 |
| I/O + 简单 | threading | GIL 影响小 |
| CPU 密集 | multiprocessing | 绑各自 GIL |
| 混合 | 进程池 + asyncio | 最佳组合 |

### import 查找顺序

```python
import sys
print(sys.path)
# 1. 当前脚本目录
# 2. PYTHONPATH 环境变量
# 3. 标准库
# 4. site-packages
```

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 多线程 CPU 密集 | GIL 导致无法并行 | 用 multiprocessing |
| 循环引用内存泄漏 | 引用计数无法释放 | 避免或用弱引用 |
| 相对导入失败 | 直接运行模块 | 用 `python -m` |
| .pyc 缓存问题 | 代码改了没生效 | 删除 `__pycache__` |

## 学习路径

1. [CPython 执行链路](docs/01-cpython-execution.md)
2. [字节码与 .pyc](docs/02-bytecode-pyc.md)
3. [GIL 与并发](docs/03-gil-concurrency.md)
4. [import 系统](docs/04-import-system.md)
5. [内存与 GC](docs/05-memory-gc.md)
6. [错误与 traceback](docs/06-traceback.md)
7. [性能工具](docs/07-performance-tools.md)
8. [面试题](docs/08-interview-questions.md)

