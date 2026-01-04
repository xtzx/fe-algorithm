# P10: 调试与性能优化

> 掌握调试技巧，能定位和解决性能问题

## 学完后能做

- 使用 pdb 调试
- 配置生产级日志
- 定位性能瓶颈

## 快速开始

```bash
# 运行示例
cd examples
python pdb_demo.py
python logging_demo.py
python profile_demo.py
python memory_demo.py
```

## Python vs JavaScript 调试对比

| 功能 | Python | JavaScript |
|------|--------|------------|
| 调试器 | pdb / breakpoint() | debugger 语句 |
| 日志 | logging 模块 | console.log / winston |
| 性能分析 | cProfile | Chrome DevTools |
| 内存分析 | tracemalloc | Chrome Memory |
| IDE 调试 | VS Code / PyCharm | Chrome DevTools / VS Code |

## 目录结构

```
py-10-debugging/
├── README.md
├── docs/
│   ├── 01-pdb.md              # pdb 调试器
│   ├── 02-logging.md          # 日志最佳实践
│   ├── 03-profiling.md        # 性能分析
│   ├── 04-memory.md           # 内存分析
│   ├── 05-optimization.md     # 优化技巧
│   ├── 06-exercises.md        # 练习题
│   └── 07-interview-questions.md # 面试题
├── examples/
│   ├── pdb_demo.py
│   ├── logging_demo.py
│   ├── profile_demo.py
│   └── memory_demo.py
└── scripts/
```

## 核心概念速查

### pdb 常用命令

| 命令 | 说明 |
|------|------|
| `n` | next - 执行下一行 |
| `s` | step - 进入函数 |
| `c` | continue - 继续执行 |
| `p expr` | print - 打印表达式 |
| `l` | list - 显示代码 |
| `w` | where - 显示调用栈 |
| `b line` | breakpoint - 设置断点 |
| `q` | quit - 退出调试 |

### 日志级别

| 级别 | 值 | 用途 |
|------|---|------|
| DEBUG | 10 | 详细调试信息 |
| INFO | 20 | 一般信息 |
| WARNING | 30 | 警告 |
| ERROR | 40 | 错误 |
| CRITICAL | 50 | 严重错误 |

### 性能分析命令

```bash
# cProfile
python -m cProfile -s cumtime script.py

# tracemalloc（内存）
python -c "import tracemalloc; tracemalloc.start(); ..."
```

## 常见性能问题

| 问题 | 优化方案 |
|------|---------|
| 字符串拼接 | 使用 `''.join()` |
| 列表追加 | 使用生成器或预分配 |
| 全局变量 | 缓存到局部变量 |
| 重复计算 | 使用 `@lru_cache` |
| 不当数据结构 | 选择合适的结构 |

## 学习路径

1. [pdb 调试器](docs/01-pdb.md)
2. [日志最佳实践](docs/02-logging.md)
3. [性能分析](docs/03-profiling.md)
4. [内存分析](docs/04-memory.md)
5. [优化技巧](docs/05-optimization.md)

