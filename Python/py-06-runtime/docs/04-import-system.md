# 04. import 系统

## 本节目标

- 理解 Python 的模块查找机制
- 掌握包与模块的区别
- 熟练使用各种导入方式

---

## 模块查找顺序

当执行 `import module` 时，Python 按以下顺序查找：

```python
import sys
print(sys.path)

# 1. 当前脚本所在目录
# 2. PYTHONPATH 环境变量中的目录
# 3. 标准库目录
# 4. site-packages（第三方包）
```

### sys.path 示例

```python
import sys

# 查看完整路径
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

# 动态添加路径
sys.path.insert(0, '/my/custom/path')
```

### 对比 Node.js

```javascript
// Node.js 查找顺序
// 1. 核心模块
// 2. node_modules（逐级向上）
// 3. 全局 node_modules
```

---

## 包与模块

### 模块 (Module)

单个 `.py` 文件就是一个模块。

```python
# mymodule.py
def hello():
    return "Hello"

# 使用
import mymodule
mymodule.hello()
```

### 包 (Package)

包含 `__init__.py` 的目录。

```
mypackage/
├── __init__.py
├── module1.py
└── subpackage/
    ├── __init__.py
    └── module2.py
```

```python
# 导入包
import mypackage
from mypackage import module1
from mypackage.subpackage import module2
```

### `__init__.py` 的作用

```python
# mypackage/__init__.py

# 1. 标识这是一个包（Python 3.3+ 可选）

# 2. 包级别的初始化代码
print("包被导入了")

# 3. 定义 __all__ 控制 from package import *
__all__ = ['module1', 'module2']

# 4. 暴露子模块内容
from .module1 import func1
from .module2 import func2
```

---

## 导入方式

### 绝对导入

```python
# 从包的根目录开始
from mypackage.subpackage import module
from mypackage.subpackage.module import func

import mypackage.subpackage.module as mod
```

### 相对导入

```python
# 只能在包内使用
# 假设当前文件是 mypackage/subpackage/module2.py

from . import module1           # 同级目录
from .module1 import func       # 同级模块的函数
from .. import other_module     # 上级目录
from ..other import something   # 上级的兄弟包
```

### 相对导入 vs 绝对导入

| 特性 | 绝对导入 | 相对导入 |
|------|----------|----------|
| 语法 | `from pkg.mod import x` | `from .mod import x` |
| 适用 | 任何地方 | 只能在包内 |
| 可读性 | 更清晰 | 更简洁 |
| 重构 | 改动多 | 自适应 |
| 推荐 | 一般推荐 | 包内推荐 |

---

## `python -m` 运行方式

```bash
# 直接运行脚本
python script.py

# 作为模块运行
python -m module_name

# 区别
python app/main.py      # __name__ = "__main__", 不识别包结构
python -m app.main      # __name__ = "__main__", 识别包结构
```

### 为什么用 `-m`

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── utils.py
```

```python
# app/main.py
from .utils import helper  # 相对导入

# 直接运行会报错
# python app/main.py  # ImportError

# 正确方式
# python -m app.main  # 正常工作
```

---

## `__main__.py`

让包可以直接运行：

```
mypackage/
├── __init__.py
├── __main__.py
└── core.py
```

```python
# mypackage/__main__.py
from .core import main

if __name__ == "__main__":
    main()
```

```bash
# 可以这样运行
python -m mypackage
```

---

## 命名空间包

Python 3.3+ 支持没有 `__init__.py` 的包：

```
# 分布在不同位置的同名包会合并
/path1/namespace_pkg/
    module1.py

/path2/namespace_pkg/
    module2.py

# 如果 /path1 和 /path2 都在 sys.path 中
import namespace_pkg.module1  # 从 /path1
import namespace_pkg.module2  # 从 /path2
```

用于：
- 大型项目拆分
- 插件系统

---

## 动态导入

### importlib

```python
import importlib

# 动态导入模块
module = importlib.import_module('json')
print(module.dumps({'a': 1}))

# 动态导入子模块
submodule = importlib.import_module('os.path')

# 重新加载模块（热更新）
importlib.reload(module)
```

### `__import__`

```python
# 底层函数（通常用 importlib 代替）
module = __import__('json')
```

---

## 导入钩子

### 自定义导入器

```python
import sys
from importlib.abc import Finder, Loader
from importlib.machinery import ModuleSpec

class MyFinder(Finder):
    def find_spec(self, name, path, target=None):
        if name == "virtual_module":
            return ModuleSpec(name, MyLoader())
        return None

class MyLoader(Loader):
    def create_module(self, spec):
        return None  # 使用默认创建

    def exec_module(self, module):
        module.message = "I'm virtual!"

# 注册
sys.meta_path.insert(0, MyFinder())

# 使用
import virtual_module
print(virtual_module.message)
```

---

## 常见问题

### 循环导入

```python
# a.py
from b import func_b
def func_a(): pass

# b.py
from a import func_a  # ImportError: 循环导入
def func_b(): pass
```

**解决方案**：

```python
# 1. 延迟导入
def func_b():
    from a import func_a
    func_a()

# 2. 重构代码结构
# 3. 使用 TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a import TypeA
```

### 相对导入失败

```bash
# 错误
python package/module.py
# ImportError: attempted relative import with no known parent package

# 正确
python -m package.module
```

### 模块不是包

```python
# 错误：尝试从模块导入子模块
from json.encoder import something
# 只有当 json 是包（目录）时才行
# json 是单文件模块时会失败
```

---

## `__name__` 和 `__main__`

```python
# mymodule.py
print(f"__name__ = {__name__}")

def main():
    print("Running as main")

if __name__ == "__main__":
    main()
```

```bash
python mymodule.py
# __name__ = __main__
# Running as main

python -c "import mymodule"
# __name__ = mymodule
```

---

## 本节要点

1. **sys.path**: 模块查找路径列表
2. **包 vs 模块**: 目录 vs 单文件
3. **`__init__.py`**: 包的入口和配置
4. **相对导入**: 只能在包内，用 `.` 和 `..`
5. **`python -m`**: 以模块方式运行，识别包结构
6. **`__main__.py`**: 让包可以直接运行
7. **importlib**: 动态导入和重新加载

