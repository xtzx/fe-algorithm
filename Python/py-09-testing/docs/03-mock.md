# 03. Mock 策略

## 本节目标

- 理解 mock 的作用和边界
- 掌握 unittest.mock 的使用
- 知道何时应该/不应该 mock

---

## 什么是 Mock

Mock 是替换真实对象的虚假对象：

```
类比 JavaScript:
unittest.mock ≈ jest.mock / jest.fn
```

**用途**：
- 隔离外部依赖（网络、数据库）
- 控制返回值
- 验证调用

---

## 基本 Mock

```python
from unittest.mock import Mock, MagicMock

# 创建 mock 对象
mock = Mock()

# 设置返回值
mock.return_value = 42
assert mock() == 42

# 设置属性
mock.name = "test"
assert mock.name == "test"

# 设置方法返回值
mock.method.return_value = "hello"
assert mock.method() == "hello"

# 验证调用
mock.method()
mock.method.assert_called()
mock.method.assert_called_once()
```

---

## patch - 替换模块内容

### 装饰器方式

```python
from unittest.mock import patch

# 假设 mymodule.py 中使用了 requests.get
@patch("mymodule.requests.get")
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {"data": "test"}

    result = mymodule.fetch_data()

    assert result == {"data": "test"}
    mock_get.assert_called_once()
```

### 上下文管理器方式

```python
def test_api_call():
    with patch("mymodule.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"data": "test"}

        result = mymodule.fetch_data()

        assert result == {"data": "test"}
```

### 重要：patch 的路径

```python
# mymodule.py
from requests import get  # 导入到 mymodule 命名空间

def fetch_data():
    return get("http://api.com").json()

# test_mymodule.py
# ❌ 错误：patch 原始位置
@patch("requests.get")

# ✓ 正确：patch 使用位置
@patch("mymodule.get")
```

**规则**：patch 的是**使用处**，而非定义处。

---

## MagicMock

MagicMock 支持魔术方法：

```python
from unittest.mock import MagicMock

# 支持上下文管理器
mock = MagicMock()
with mock as m:
    pass
mock.__enter__.assert_called()

# 支持迭代
mock.__iter__.return_value = iter([1, 2, 3])
assert list(mock) == [1, 2, 3]

# 支持长度
mock.__len__.return_value = 5
assert len(mock) == 5
```

---

## 验证调用

```python
from unittest.mock import Mock, call

mock = Mock()

# 调用
mock("arg1", key="value")
mock("arg2")

# 验证
mock.assert_called()                           # 被调用过
mock.assert_called_once()                      # 只调用一次（失败）
mock.assert_called_with("arg2")                # 最后一次调用参数
mock.assert_any_call("arg1", key="value")      # 任意调用匹配

# 验证调用次数
assert mock.call_count == 2

# 验证调用顺序
mock.assert_has_calls([
    call("arg1", key="value"),
    call("arg2"),
])
```

---

## side_effect - 动态行为

```python
from unittest.mock import Mock

# 抛出异常
mock = Mock(side_effect=ValueError("error"))
# mock() 会抛出 ValueError

# 返回不同值
mock = Mock(side_effect=[1, 2, 3])
assert mock() == 1
assert mock() == 2
assert mock() == 3

# 自定义函数
def custom_side_effect(x):
    if x < 0:
        raise ValueError("negative")
    return x * 2

mock = Mock(side_effect=custom_side_effect)
assert mock(5) == 10
```

---

## 时间和随机数控制

### 控制时间

```python
from unittest.mock import patch
import time

@patch("time.time")
def test_with_fixed_time(mock_time):
    mock_time.return_value = 1000.0
    assert time.time() == 1000.0

# 使用 freezegun 库更方便
from freezegun import freeze_time

@freeze_time("2024-01-01 12:00:00")
def test_frozen_time():
    from datetime import datetime
    assert datetime.now().year == 2024
```

### 控制随机数

```python
from unittest.mock import patch

@patch("random.randint")
def test_random(mock_randint):
    mock_randint.return_value = 42
    import random
    assert random.randint(1, 100) == 42
```

---

## pytest-mock

pytest-mock 提供更方便的 fixture：

```python
def test_with_mocker(mocker):
    # 替代 @patch
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.json.return_value = {"data": "test"}

    # spy：保持原实现但可验证
    spy = mocker.spy(obj, "method")
    obj.method()
    spy.assert_called_once()

    # stub：简单替换
    mocker.stub(name="stub")
```

---

## Mock 边界原则

### 应该 Mock

1. **外部服务**：HTTP API、数据库、消息队列
2. **文件系统**：写入文件、读取配置
3. **时间相关**：当前时间、延迟
4. **随机数**：需要确定性结果

### 不应该 Mock

1. **被测代码本身**：失去测试意义
2. **简单依赖**：内存中的对象
3. **过多 mock**：可能是设计问题

### 原则

```
Mock 在边界：
  ✓ 你的代码 <-> 外部系统
  ✗ 你的代码 <-> 你的代码
```

---

## 测试异步函数

```python
import pytest
from unittest.mock import AsyncMock, patch

async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://api.com")
        return response.json()

@pytest.mark.asyncio
async def test_async_function():
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_instance
        mock_instance.get.return_value.json.return_value = {"data": "test"}

        result = await fetch_data()

        assert result == {"data": "test"}
```

---

## 与 Jest 对比

**Jest:**
```javascript
jest.mock('axios');
axios.get.mockResolvedValue({ data: { result: 'test' } });

test('fetch data', async () => {
    const result = await fetchData();
    expect(axios.get).toHaveBeenCalled();
});
```

**pytest:**
```python
@patch("requests.get")
def test_fetch_data(mock_get):
    mock_get.return_value.json.return_value = {"result": "test"}

    result = fetch_data()

    mock_get.assert_called_once()
```

---

## 本节要点

1. **Mock** 替换真实对象
2. **patch** 替换模块内容（注意路径）
3. **MagicMock** 支持魔术方法
4. **side_effect** 实现动态行为
5. **边界原则**：只 mock 外部依赖
6. **AsyncMock** 处理异步

