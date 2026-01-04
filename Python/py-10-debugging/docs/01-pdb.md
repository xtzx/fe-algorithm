# 01. pdb 调试器

## 本节目标

- 掌握 pdb 的基本使用
- 学会条件断点和事后调试
- 配置 VS Code 调试

---

## breakpoint() - Python 3.7+

最简单的调试方式：

```python
def calculate(x, y):
    result = x + y
    breakpoint()  # 在这里暂停
    return result * 2

calculate(3, 4)
```

运行后进入交互式调试器。

---

## pdb 常用命令

### 执行控制

| 命令 | 全名 | 说明 |
|------|------|------|
| `n` | next | 执行下一行（不进入函数） |
| `s` | step | 进入函数内部 |
| `c` | continue | 继续执行到下一个断点 |
| `r` | return | 执行到当前函数返回 |
| `q` | quit | 退出调试器 |

### 查看信息

| 命令 | 全名 | 说明 |
|------|------|------|
| `p expr` | print | 打印表达式的值 |
| `pp expr` | pretty-print | 美化打印 |
| `l` | list | 显示当前代码 |
| `ll` | long list | 显示整个函数 |
| `w` | where | 显示调用栈 |
| `a` | args | 显示当前函数参数 |

### 断点操作

| 命令 | 说明 |
|------|------|
| `b lineno` | 在指定行设置断点 |
| `b func` | 在函数入口设置断点 |
| `b file:lineno` | 在指定文件的行设置断点 |
| `b` | 列出所有断点 |
| `cl num` | 清除指定断点 |
| `disable num` | 禁用断点 |
| `enable num` | 启用断点 |

---

## 启动调试的方式

### 1. breakpoint()

```python
def buggy_function():
    x = 1
    breakpoint()
    y = x / 0  # 这里会出错
```

### 2. python -m pdb

```bash
python -m pdb script.py
```

### 3. import pdb

```python
import pdb

def buggy_function():
    x = 1
    pdb.set_trace()  # 老式写法
    y = x / 0
```

### 4. 事后调试（Post-mortem）

```bash
# 程序崩溃后进入调试
python -m pdb -c continue script.py
```

或在代码中：

```python
import pdb
import sys

def main():
    # 你的代码
    pass

if __name__ == "__main__":
    try:
        main()
    except:
        pdb.post_mortem()
```

---

## 条件断点

```python
# 在 pdb 中
(Pdb) b 10, x > 5  # 当 x > 5 时在第 10 行暂停
(Pdb) b func, len(items) == 0  # 当 items 为空时在 func 入口暂停
```

---

## 调试示例

```python
def find_bug():
    items = [1, 2, 3, 4, 5]
    total = 0

    for i, item in enumerate(items):
        breakpoint()  # 暂停查看状态
        total += item
        print(f"i={i}, item={item}, total={total}")

    return total

find_bug()
```

调试会话：
```
> find_bug()
-> total += item
(Pdb) p i
0
(Pdb) p item
1
(Pdb) p total
0
(Pdb) n
> print(f"i={i}...")
(Pdb) c
> total += item
(Pdb) p i
1
(Pdb) q
```

---

## VS Code 调试配置

### launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: 当前文件",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Python: 模块",
            "type": "debugpy",
            "request": "launch",
            "module": "myapp",
            "console": "integratedTerminal"
        },
        {
            "name": "Python: pytest",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

### 使用方法

1. 在代码中点击行号左侧设置断点
2. 按 F5 启动调试
3. 使用调试工具栏控制执行

---

## 禁用 breakpoint()

```bash
# 环境变量禁用所有 breakpoint()
PYTHONBREAKPOINT=0 python script.py

# 使用自定义调试器
PYTHONBREAKPOINT=ipdb.set_trace python script.py
```

---

## 远程调试

### debugpy（VS Code）

```python
import debugpy

# 等待调试器连接
debugpy.listen(("0.0.0.0", 5678))
print("等待调试器连接...")
debugpy.wait_for_client()

# 你的代码
```

VS Code launch.json：
```json
{
    "name": "Python: 远程附加",
    "type": "debugpy",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 5678
    }
}
```

---

## ipdb - 增强版 pdb

```bash
pip install ipdb
```

```python
import ipdb
ipdb.set_trace()

# 或设置环境变量
# PYTHONBREAKPOINT=ipdb.set_trace python script.py
```

**优势**：
- 语法高亮
- Tab 补全
- 更好的上下文显示

---

## 调试技巧

### 打印变量

```python
(Pdb) p locals()  # 所有局部变量
(Pdb) p dir(obj)  # 对象属性
(Pdb) p type(x)   # 变量类型
```

### 执行代码

```python
(Pdb) !x = 10  # 修改变量
(Pdb) !import pprint; pprint.pprint(data)
```

### 跳转

```python
(Pdb) j 20  # 跳转到第 20 行（危险，可能导致未定义行为）
```

---

## 本节要点

1. **breakpoint()** 是 Python 3.7+ 推荐的调试方式
2. **n/s/c** 控制执行流程
3. **p/l/w** 查看状态
4. **条件断点** `b line, condition`
5. **事后调试** `pdb.post_mortem()`
6. **VS Code** 提供图形化调试

