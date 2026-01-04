# P01: Python 基础语法

> 面向 JS/TS 资深工程师的 Python 入门教程

## 🎯 学完后能做

- ✅ 读懂 Python 代码
- ✅ 写出基本的 Python 脚本
- ✅ 理解 Python 与 JS 的核心差异

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 检查 Python 版本（需要 3.12+）
python3 --version

# 或者使用 pyenv 安装
pyenv install 3.12.0
pyenv global 3.12.0
```

### 2. 运行示例

```bash
# 进入示例目录
cd examples

# 运行 Hello World
python3 01_hello.py

# 运行所有示例
cd ../scripts && bash run_all.sh
```

### 3. REPL 交互

```bash
# 进入 Python 交互式环境
python3

>>> print("Hello, Python!")
>>> exit()
```

---

## 📚 目录结构

```
py-01-basics/
├── README.md               # 本文件
├── docs/                   # 教程文档
│   ├── 01-environment-setup.md
│   ├── 02-variables-and-types.md
│   ├── 03-operators.md
│   ├── 04-strings.md
│   ├── 05-control-flow.md
│   ├── 06-functions.md
│   ├── 07-file-io.md
│   ├── 08-js-comparison-table.md
│   ├── 09-exercises.md
│   └── 10-interview-questions.md
├── examples/               # 示例代码
├── exercises/              # 练习题
│   ├── basic/
│   ├── advanced/
│   └── challenge/
├── project/                # 小项目
│   └── text_analyzer/
└── scripts/
    └── run_all.sh
```

---

## ⚡ Python vs JavaScript 核心差异速查

| 特性 | Python | JavaScript |
|------|--------|------------|
| **缩进** | 强制缩进（语法） | 可选（风格） |
| **分号** | 不需要 | 可选（推荐不加） |
| **变量声明** | 直接赋值 `x = 1` | `let/const x = 1` |
| **常量** | 无关键字（约定 `UPPER_CASE`） | `const` |
| **None/null** | `None` | `null` / `undefined` |
| **布尔值** | `True` / `False` | `true` / `false` |
| **逻辑运算** | `and` / `or` / `not` | `&&` / `\|\|` / `!` |
| **整除** | `//` | `Math.floor(a/b)` |
| **幂运算** | `**` | `**` |
| **字符串模板** | `f"Hello {name}"` | `` `Hello ${name}` `` |
| **三元表达式** | `x if cond else y` | `cond ? x : y` |
| **for 循环** | `for x in list:` | `for (x of list)` |
| **函数定义** | `def func():` | `function func()` |
| **类定义** | `class Foo:` | `class Foo {}` |
| **导入** | `import / from x import y` | `import / require` |
| **类型检查** | `type()` / `isinstance()` | `typeof` / `instanceof` |

---

## 🔥 Python 独特概念

### 1. 缩进即语法

```python
# Python：缩进决定代码块
if True:
    print("Hello")  # 4 空格缩进
    print("World")

# JavaScript：大括号决定代码块
# if (true) {
#     console.log("Hello");
# }
```

### 2. 切片操作

```python
s = "Hello World"
s[0]      # 'H'（第一个字符）
s[-1]     # 'd'（最后一个字符）
s[0:5]    # 'Hello'（索引 0-4）
s[::2]    # 'HloWrd'（每隔一个）
s[::-1]   # 'dlroW olleH'（反转）
```

### 3. 多返回值

```python
def get_point():
    return 10, 20

x, y = get_point()  # 元组解包
```

### 4. 列表推导式

```python
# Python
squares = [x**2 for x in range(10)]

# JavaScript
# const squares = [...Array(10)].map((_, x) => x ** 2);
```

---

## ⚠️ 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| **缩进错误** | 混用 Tab 和空格 | 统一使用 4 空格 |
| **可变默认参数** | `def f(lst=[])` 会共享 | `def f(lst=None)` |
| **is vs ==** | `[] is []` 为 `False` | 比较值用 `==` |
| **整数除法** | `3 / 2 = 1.5` | 整除用 `//` |
| **字符串不可变** | `s[0] = 'a'` 报错 | `s = 'a' + s[1:]` |

---

## 📖 学习路径

1. [环境配置](docs/01-environment-setup.md)
2. [变量与类型](docs/02-variables-and-types.md)
3. [运算符](docs/03-operators.md)
4. [字符串](docs/04-strings.md)
5. [控制流](docs/05-control-flow.md)
6. [函数](docs/06-functions.md)
7. [文件 I/O](docs/07-file-io.md)
8. [JS 对照表](docs/08-js-comparison-table.md)
9. [练习题](docs/09-exercises.md)
10. [面试题](docs/10-interview-questions.md)

---

## 🛠️ 小项目：文本统计器

完成后，尝试实现 `project/text_analyzer/`：

```bash
python3 project/text_analyzer/main.py sample.txt
```

输出：
```
Lines: 42
Words: 256
Characters: 1832
Longest line: 78 chars
```

---

## 📝 License

MIT

