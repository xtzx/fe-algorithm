# 04. 覆盖率

## 本节目标

- 理解代码覆盖率的意义
- 使用 coverage.py 和 pytest-cov
- 设置合理的覆盖率目标

---

## 什么是覆盖率

代码覆盖率衡量测试执行了多少代码：

```
类比 JavaScript:
coverage.py ≈ istanbul (nyc)
pytest-cov ≈ jest --coverage
```

---

## 覆盖率类型

| 类型 | 说明 |
|------|------|
| 行覆盖率 | 执行了多少行代码 |
| 分支覆盖率 | 执行了多少分支（if/else） |
| 函数覆盖率 | 调用了多少函数 |
| 语句覆盖率 | 执行了多少语句 |

---

## 使用 pytest-cov

### 安装

```bash
pip install pytest-cov
```

### 基本使用

```bash
# 运行测试并收集覆盖率
pytest --cov=src

# 显示缺失行
pytest --cov=src --cov-report=term-missing

# 生成 HTML 报告
pytest --cov=src --cov-report=html

# 生成 XML 报告（CI 用）
pytest --cov=src --cov-report=xml
```

### 输出示例

```
---------- coverage: platform darwin, python 3.12 -----------
Name                      Stmts   Miss  Cover   Missing
---------------------------------------------------------
src/math_utils.py            20      2    90%   15-16
src/file_utils.py            35     10    71%   25-30, 42-46
src/http_utils.py            15      0   100%
---------------------------------------------------------
TOTAL                        70     12    83%
```

---

## 配置

### pyproject.toml

```toml
[tool.coverage.run]
# 源代码目录
source = ["src"]

# 启用分支覆盖
branch = true

# 忽略
omit = [
    "*/tests/*",
    "*/__init__.py",
]

# 并发模式
concurrency = ["thread", "multiprocessing"]

[tool.coverage.report]
# 排除行
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]

# 显示缺失行
show_missing = true

# 最小覆盖率
fail_under = 80

# 精度
precision = 2

[tool.coverage.html]
directory = "htmlcov"
```

### pytest.ini / pyproject.toml

```toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=term-missing"
```

---

## 忽略代码

### 行内忽略

```python
if DEBUG:  # pragma: no cover
    print("debug mode")
```

### 块忽略

```python
if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional
```

### 配置忽略

```toml
[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
]
```

---

## 分支覆盖

```python
def example(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
```

**行覆盖率**：只需调用 `example(1)` 就能覆盖所有行。

**分支覆盖率**：需要同时调用 `example(1)` 和 `example(-1)` 才能覆盖所有分支。

```bash
pytest --cov=src --cov-branch
```

---

## 覆盖率目标

### 建议

| 目标 | 说明 |
|------|------|
| 80% | 合理目标 |
| 90%+ | 严格项目 |
| 100% | 不推荐作为硬性目标 |

### 100% 覆盖率的问题

1. **边际效益递减**：最后 10% 成本很高
2. **虚假安全感**：覆盖不等于正确
3. **测试质量下降**：为了覆盖而写无意义测试

### 正确态度

```
高覆盖率 ≠ 高质量
低覆盖率 = 可能有问题
```

---

## CI 集成

### GitHub Actions

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### 覆盖率徽章

使用 Codecov 或 Coveralls 生成徽章：

```markdown
[![codecov](https://codecov.io/gh/user/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/user/repo)
```

---

## 覆盖率差异

只检查变更代码的覆盖率：

```bash
# 与 main 分支比较
diff-cover coverage.xml --compare-branch=main
```

---

## 常用命令

```bash
# 运行并收集覆盖率
pytest --cov=src

# 显示缺失行
pytest --cov=src --cov-report=term-missing

# 生成 HTML 报告
pytest --cov=src --cov-report=html
open htmlcov/index.html

# 设置最小覆盖率
pytest --cov=src --cov-fail-under=80

# 清除缓存
coverage erase
```

---

## 本节要点

1. **覆盖率** 衡量测试覆盖的代码量
2. **分支覆盖** 比行覆盖更严格
3. **80%** 是合理的覆盖率目标
4. **100% 不是目标**：质量比数量重要
5. **CI 集成**：自动检查覆盖率
6. 使用 **`# pragma: no cover`** 排除代码

