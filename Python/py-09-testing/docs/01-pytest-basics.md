# 01. pytest 基础

## 本节目标

- 理解 pytest 的测试发现机制
- 掌握基本断言和参数化
- 对比 JavaScript 测试框架

---

## pytest vs Jest

| 特性 | pytest | Jest |
|------|--------|------|
| 断言 | `assert x == y` | `expect(x).toBe(y)` |
| 测试函数 | `def test_xxx():` | `test('xxx', () => {})` |
| 参数化 | `@pytest.mark.parametrize` | `test.each` |
| 设置/清理 | fixture | beforeEach/afterEach |
| Mock | unittest.mock | jest.mock |

---

## 测试发现

pytest 自动发现测试：

```
tests/
├── test_*.py          # 文件名以 test_ 开头
└── *_test.py          # 或以 _test 结尾

# 文件内部
def test_xxx():        # 函数以 test_ 开头
class TestXxx:         # 类以 Test 开头
    def test_yyy():    # 方法以 test_ 开头
```

---

## 基本断言

pytest 使用原生 `assert` 语句：

```python
def test_equality():
    assert 1 + 1 == 2

def test_in():
    assert "hello" in "hello world"

def test_instance():
    assert isinstance([], list)

def test_exception():
    with pytest.raises(ValueError):
        int("not a number")

def test_exception_message():
    with pytest.raises(ValueError, match="invalid literal"):
        int("abc")
```

### 断言信息

```python
def test_with_message():
    x = 1
    assert x == 2, f"Expected 2, got {x}"
```

---

## 参数化测试

### 基本参数化

```python
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add(a, b, expected):
    assert a + b == expected
```

### 多参数组合

```python
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_multiply(x, y):
    # 测试 (1,10), (1,20), (2,10), (2,20)
    assert x * y > 0
```

### 带 ID 的参数化

```python
@pytest.mark.parametrize("input,expected", [
    pytest.param("hello", 5, id="simple"),
    pytest.param("", 0, id="empty"),
    pytest.param("世界", 2, id="unicode"),
])
def test_length(input, expected):
    assert len(input) == expected
```

---

## 测试异常

```python
import pytest

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# 测试异常类型
def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(1, 0)

# 测试异常消息
def test_divide_by_zero_message():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(1, 0)

# 获取异常对象
def test_exception_info():
    with pytest.raises(ValueError) as exc_info:
        divide(1, 0)
    assert "zero" in str(exc_info.value)
```

---

## 跳过测试

```python
import pytest
import sys

# 无条件跳过
@pytest.mark.skip(reason="尚未实现")
def test_not_implemented():
    pass

# 条件跳过
@pytest.mark.skipif(sys.platform == "win32", reason="不支持 Windows")
def test_unix_only():
    pass

# 预期失败
@pytest.mark.xfail(reason="已知 bug")
def test_known_bug():
    assert 1 == 2
```

---

## 命令行选项

```bash
# 运行所有测试
pytest

# 详细输出
pytest -v

# 运行特定文件
pytest tests/test_math.py

# 运行特定测试
pytest tests/test_math.py::test_add

# 按名称匹配
pytest -k "add or subtract"

# 失败即停
pytest -x

# 最后 N 个失败
pytest --lf  # 只运行上次失败的
pytest --ff  # 先运行上次失败的

# 显示打印输出
pytest -s

# 调试模式
pytest --pdb
```

---

## 测试类

```python
class TestCalculator:
    """计算器测试类"""

    def test_add(self):
        assert 1 + 1 == 2

    def test_subtract(self):
        assert 5 - 3 == 2

    def test_multiply(self):
        assert 3 * 4 == 12
```

---

## 与 unittest 对比

| 特性 | pytest | unittest |
|------|--------|----------|
| 断言 | `assert x == y` | `self.assertEqual(x, y)` |
| 设置 | fixture | setUp/tearDown |
| 参数化 | 内置 | 需要第三方库 |
| 发现 | 自动 | 需要显式 |
| 输出 | 更友好 | 基础 |

```python
# unittest 风格
import unittest

class TestMath(unittest.TestCase):
    def setUp(self):
        self.calculator = Calculator()

    def test_add(self):
        self.assertEqual(self.calculator.add(1, 2), 3)

    def tearDown(self):
        pass

# pytest 风格
def test_add():
    assert 1 + 2 == 3
```

---

## 本节要点

1. **测试发现**: test_*.py 文件，test_ 开头的函数
2. **断言**: 使用原生 `assert`
3. **参数化**: `@pytest.mark.parametrize`
4. **异常测试**: `pytest.raises()`
5. **跳过测试**: `@pytest.mark.skip/skipif/xfail`
6. 比 unittest 更简洁、更强大

