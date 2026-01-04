# 06. 练习题

## 练习 1：配置 Ruff

创建一个项目并配置 ruff：

1. 创建 pyproject.toml
2. 启用 E、F、I、UP、B 规则
3. 配置 import 排序
4. 运行并修复问题

```toml
# 完成配置
[tool.ruff]
# ...

[tool.ruff.lint]
# ...
```

---

## 练习 2：修复代码问题

修复以下代码的所有 ruff 问题：

```python
import os
import json
from pathlib import Path
import sys

def process(x,y):
    result=x+y
    unused_var = 123
    return result

class myClass:
    def __init__(self,name):
        self.name=name
```

---

## 练习 3：配置 Pre-commit

创建 .pre-commit-config.yaml：

1. 添加 ruff (lint + format)
2. 添加 pyright
3. 添加 trailing-whitespace
4. 安装并测试

---

## 练习 4：类型注解

为以下函数添加完整的类型注解：

```python
def fetch_users(ids, include_inactive=False):
    users = []
    for id in ids:
        user = get_user(id)
        if user and (include_inactive or user.active):
            users.append(user)
    return users

def process_data(data, transform=None):
    if transform:
        return transform(data)
    return data
```

---

## 练习 5：Pyright 配置

配置 pyright：

1. 使用 standard 模式
2. 忽略 tests/ 目录
3. 设置 Python 版本为 3.12
4. 启用 reportUnusedImport

---

## 练习 6：Makefile

创建 Makefile 包含以下任务：

- install: 安装依赖
- test: 运行测试
- lint: 运行 ruff + pyright
- format: 格式化代码
- clean: 清理构建产物
- all: 默认任务（lint + test）

---

## 练习 7：处理第三方库类型

处理以下场景：

1. requests 库缺少类型
2. 安装类型桩
3. 配置 pyright 忽略缺失类型
4. 使用 type: ignore 忽略特定行

---

## 练习 8：CI 配置

创建 GitHub Actions 配置：

1. 运行 ruff check
2. 运行 ruff format --check
3. 运行 pyright
4. 运行 pytest
5. 缓存依赖

```yaml
# .github/workflows/ci.yml
# 完成配置
```

