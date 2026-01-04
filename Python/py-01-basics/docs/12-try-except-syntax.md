# try-except 完整语法

> 掌握 Python 异常处理的完整语法

## 基础语法

```python
try:
    # 可能出错的代码
    risky_operation()
except SomeException:
    # 异常处理代码
    handle_error()
```

---

## 完整四件套：try-except-else-finally

```python
try:
    # 1. 尝试执行的代码
    result = risky_operation()
except SomeException as e:
    # 2. 发生异常时执行
    handle_error(e)
else:
    # 3. 没有异常时执行（可选）
    process_result(result)
finally:
    # 4. 无论如何都执行（可选）
    cleanup()
```

### 执行顺序

```
无异常：try → else → finally
有异常：try → except → finally
```

### 各部分的作用

| 部分 | 执行时机 | 典型用途 |
|------|---------|---------|
| `try` | 总是执行 | 放置可能出错的代码 |
| `except` | 发生异常时 | 处理异常 |
| `else` | 无异常时 | 处理成功逻辑 |
| `finally` | 总是执行 | 清理资源 |

### 完整示例

```python
def read_config(path: str) -> dict:
    """读取配置文件"""
    file = None
    try:
        file = open(path, "r")
        content = file.read()
        config = json.loads(content)
    except FileNotFoundError:
        print(f"Config file not found: {path}")
        config = {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        config = {}
    else:
        # 成功读取，记录日志
        print(f"Loaded config from {path}")
    finally:
        # 确保文件被关闭
        if file:
            file.close()

    return config
```

---

## except 语法详解

### 1. 捕获单个异常

```python
try:
    value = int(input("Enter a number: "))
except ValueError:
    print("Invalid number")
```

### 2. 捕获异常并获取信息

```python
try:
    value = int("abc")
except ValueError as e:
    print(f"Error: {e}")           # Error: invalid literal for int()...
    print(f"Type: {type(e)}")      # Type: <class 'ValueError'>
    print(f"Args: {e.args}")       # Args: ("invalid literal...",)
```

### 3. 捕获多个异常（元组）

```python
try:
    result = process(data)
except (ValueError, TypeError, KeyError) as e:
    print(f"Data error: {e}")
```

### 4. 多个 except 块

```python
try:
    result = data[key] / value
except KeyError:
    print("Key not found")
except ZeroDivisionError:
    print("Cannot divide by zero")
except TypeError:
    print("Type mismatch")
```

### 5. 异常处理的顺序

```python
# ❌ 错误：父类在前会拦截子类
try:
    open("missing.txt")
except OSError:
    print("OS error")
except FileNotFoundError:  # 永远不会执行！
    print("File not found")

# ✅ 正确：子类在前
try:
    open("missing.txt")
except FileNotFoundError:
    print("File not found")  # 会执行这个
except OSError:
    print("Other OS error")
```

### 6. 捕获所有异常

```python
# 方式一：捕获 Exception（推荐）
try:
    risky_operation()
except Exception as e:
    print(f"Unexpected error: {e}")

# 方式二：裸 except（不推荐）
try:
    risky_operation()
except:  # 等同于 except BaseException
    print("Something went wrong")
```

---

## else 子句

### 为什么需要 else？

```python
# ❌ 没有 else，成功逻辑混在 try 中
try:
    file = open(path)
    content = file.read()  # 这行出错也会被捕获
    process(content)       # 这行出错也会被捕获
except FileNotFoundError:
    print("File not found")

# ✅ 使用 else，只捕获预期的异常
try:
    file = open(path)
except FileNotFoundError:
    print("File not found")
else:
    content = file.read()  # 这行出错会抛出，不会被上面的 except 捕获
    process(content)
```

### else 的使用场景

```python
# 场景：区分"尝试操作"和"后续处理"
try:
    conn = database.connect()  # 可能失败
except ConnectionError:
    use_fallback()
else:
    # 连接成功后的操作
    data = conn.query("SELECT * FROM users")
    conn.close()
```

---

## finally 子句

### finally 总是执行

```python
def example():
    try:
        return "from try"
    finally:
        print("finally executed")  # 会执行！

result = example()
# 输出: finally executed
# result = "from try"
```

### finally 与 return

```python
# finally 中的 return 会覆盖 try/except 中的 return
def bad_example():
    try:
        return 1
    finally:
        return 2  # ❌ 不推荐！

print(bad_example())  # 2

# ✅ 正确做法：finally 中不要 return
def good_example():
    result = None
    try:
        result = 1
    finally:
        cleanup()
    return result
```

### finally 的典型用途

```python
# 1. 关闭文件
file = None
try:
    file = open("data.txt")
    process(file.read())
finally:
    if file:
        file.close()

# 2. 释放锁
lock.acquire()
try:
    critical_section()
finally:
    lock.release()

# 3. 数据库连接
conn = db.connect()
try:
    conn.execute(query)
except Exception:
    conn.rollback()
    raise
else:
    conn.commit()
finally:
    conn.close()
```

---

## 重新抛出异常

### raise（不带参数）

```python
try:
    risky_operation()
except Exception as e:
    log_error(e)  # 记录日志
    raise         # 重新抛出相同异常
```

### raise 新异常

```python
try:
    value = int(user_input)
except ValueError as e:
    # 转换为更具体的异常
    raise InvalidInputError(f"Invalid input: {user_input}") from e
```

---

## JS 对照：try-catch-finally

| Python | JavaScript | 说明 |
|--------|------------|------|
| `try:` | `try {` | 尝试块 |
| `except E as e:` | `catch (e) {` | 捕获块 |
| `else:` | 无 | Python 独有 |
| `finally:` | `finally {` | 清理块 |

```javascript
// JavaScript
try {
    riskyOperation();
} catch (e) {
    if (e instanceof TypeError) {
        handleTypeError(e);
    } else {
        throw e;  // 重新抛出
    }
} finally {
    cleanup();
}
```

```python
# Python
try:
    risky_operation()
except TypeError as e:
    handle_type_error(e)
except Exception:
    raise  # 重新抛出
finally:
    cleanup()
```

### JS 没有 else，Python 的优势

```javascript
// JavaScript: 无法区分 try 中哪行出错
try {
    const file = fs.readFileSync(path);
    const data = JSON.parse(file);  // 如果这里出错...
    process(data);                   // 或这里出错...
} catch (e) {
    // 都会到这里，难以区分
}
```

```python
# Python: 精确控制
try:
    file = open(path)
except FileNotFoundError:
    handle_missing_file()
else:
    # 只有 open 成功才执行这里
    # 这里的错误不会被上面的 except 捕获
    data = json.load(file)
    process(data)
```

---

## 异常中的 return 行为

```python
def demo():
    try:
        print("try")
        return "try-return"
    except Exception:
        print("except")
        return "except-return"
    else:
        print("else")
        return "else-return"
    finally:
        print("finally")
        # 不要在这里 return！

# 无异常时
result = demo()
# 输出:
# try
# finally
# result = "try-return"

# 有异常时（假设 try 中 raise）
# 输出:
# try
# except
# finally
# result = "except-return"
```

---

## 最佳实践

### 1. 只捕获你能处理的异常

```python
# ❌ 捕获但不处理
try:
    process()
except Exception:
    pass  # 静默忽略所有错误

# ✅ 捕获并适当处理
try:
    process()
except ValueError as e:
    logger.warning(f"Invalid value: {e}")
    return default_value
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### 2. 异常信息要有用

```python
# ❌ 无用的错误信息
except Exception:
    print("Error occurred")

# ✅ 有用的错误信息
except Exception as e:
    print(f"Failed to process {filename}: {e}")
```

### 3. 优先使用 with 语句

```python
# ❌ 手动 try-finally
file = open("data.txt")
try:
    process(file.read())
finally:
    file.close()

# ✅ 使用 with（自动调用 close）
with open("data.txt") as file:
    process(file.read())
```

### 4. 避免在 finally 中 return

```python
# ❌ finally 中 return
def bad():
    try:
        return 1
    finally:
        return 2  # 覆盖了 try 的 return

# ✅ 正常结构
def good():
    try:
        result = calculate()
    finally:
        cleanup()
    return result
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 异常顺序错误 | 父类在前拦截子类 | 子类异常放前面 |
| 裸 except | 捕获了 `KeyboardInterrupt` | 用 `except Exception` |
| finally 中 return | 覆盖正常返回值 | 不要在 finally 中 return |
| 捕获后忽略 | 隐藏了错误 | 至少记录日志 |
| try 块过大 | 难以定位哪行出错 | 缩小 try 范围，用 else |

