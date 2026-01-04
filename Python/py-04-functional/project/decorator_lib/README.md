# 实用装饰器库

实现常用装饰器：`@timer`、`@retry`、`@cache`、`@validate`

## 功能

### @timer
计时装饰器，记录函数执行时间。

```python
from decorators import timer

@timer
def slow_function():
    time.sleep(1)
```

### @retry
重试装饰器，失败时自动重试。

```python
from decorators import retry

@retry(max_attempts=3, delay=1)
def unstable_function():
    # 可能失败的操作
    pass
```

### @cache
缓存装饰器，缓存函数结果。

```python
from decorators import cache

@cache
def expensive_function(n):
    return n ** 2
```

### @validate
参数验证装饰器，验证函数参数。

```python
from decorators import validate, is_positive

@validate(age=is_positive)
def create_user(name, age):
    return {"name": name, "age": age}
```

## 运行

```bash
python main.py
```

## 组合使用

装饰器可以组合使用：

```python
@timer
@cache
@retry(max_attempts=3)
def complex_function():
    pass
```

