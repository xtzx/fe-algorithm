# 描述符协议

> 理解 Python 属性访问的底层机制

## 什么是描述符

描述符是实现了 `__get__`、`__set__` 或 `__delete__` 方法的对象，用于自定义属性访问行为。

```python
class Descriptor:
    def __get__(self, obj, objtype=None):
        """获取属性时调用"""
        pass

    def __set__(self, obj, value):
        """设置属性时调用"""
        pass

    def __delete__(self, obj):
        """删除属性时调用"""
        pass
```

---

## 描述符类型

### 数据描述符（Data Descriptor）

实现了 `__get__` 和 `__set__`（或 `__delete__`）

```python
class DataDescriptor:
    def __get__(self, obj, objtype=None):
        print("__get__ called")
        return obj.__dict__.get('_value')

    def __set__(self, obj, value):
        print("__set__ called")
        obj.__dict__['_value'] = value

class MyClass:
    attr = DataDescriptor()

obj = MyClass()
obj.attr = 42      # __set__ called
print(obj.attr)    # __get__ called → 42
```

### 非数据描述符（Non-Data Descriptor）

只实现了 `__get__`

```python
class NonDataDescriptor:
    def __get__(self, obj, objtype=None):
        print("__get__ called")
        return "descriptor value"

class MyClass:
    attr = NonDataDescriptor()

obj = MyClass()
print(obj.attr)    # __get__ called → "descriptor value"

# 可以被实例属性覆盖！
obj.attr = "instance value"
print(obj.attr)    # "instance value"（不调用 __get__）
```

### 属性查找优先级

```
1. 数据描述符（__get__ + __set__）
2. 实例 __dict__
3. 非数据描述符（只有 __get__）
4. 类 __dict__
5. __getattr__（如果定义了）
```

---

## __get__ 参数详解

```python
class Descriptor:
    def __get__(self, obj, objtype=None):
        """
        参数:
            obj: 访问属性的实例，通过类访问时为 None
            objtype: 访问属性的类
        """
        if obj is None:
            # 通过类访问: MyClass.attr
            return self
        # 通过实例访问: instance.attr
        return obj.__dict__.get('_value')

class MyClass:
    attr = Descriptor()

# 通过类访问
print(MyClass.attr)        # obj=None, objtype=MyClass

# 通过实例访问
obj = MyClass()
print(obj.attr)           # obj=instance, objtype=MyClass
```

---

## 实用描述符示例

### 类型验证描述符

```python
class Typed:
    """类型验证描述符"""

    def __init__(self, name: str, expected_type: type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        obj.__dict__[self.name] = value

    def __delete__(self, obj):
        del obj.__dict__[self.name]

class Person:
    name = Typed('name', str)
    age = Typed('age', int)

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# 使用
p = Person("Alice", 30)
p.age = "thirty"  # TypeError: age must be int, got str
```

### 范围验证描述符

```python
class Range:
    """范围验证描述符"""

    def __init__(self, name: str, min_val: float, max_val: float):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if not self.min_val <= value <= self.max_val:
            raise ValueError(
                f"{self.name} must be between {self.min_val} and {self.max_val}"
            )
        obj.__dict__[self.name] = value

class Product:
    price = Range('price', 0, 10000)
    quantity = Range('quantity', 0, 1000)

    def __init__(self, price: float, quantity: int):
        self.price = price
        self.quantity = quantity
```

### 惰性计算描述符

```python
class Lazy:
    """惰性计算描述符，只计算一次"""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        # 计算值
        value = self.func(obj)

        # 存入实例 __dict__，下次直接返回（绕过描述符）
        obj.__dict__[self.name] = value

        return value

class DataProcessor:
    def __init__(self, data: list):
        self.data = data

    @Lazy
    def processed(self):
        """耗时计算，只执行一次"""
        print("Processing...")
        return [x ** 2 for x in self.data]

processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.processed)  # Processing... [1, 4, 9, 16, 25]
print(processor.processed)  # [1, 4, 9, 16, 25]（不再 Processing）
```

### 只读描述符

```python
class ReadOnly:
    """只读属性描述符"""

    def __init__(self, value):
        self.value = value

    def __get__(self, obj, objtype=None):
        return self.value

    def __set__(self, obj, value):
        raise AttributeError("Read-only attribute")

class Config:
    version = ReadOnly("1.0.0")
    app_name = ReadOnly("MyApp")

config = Config()
print(config.version)      # "1.0.0"
config.version = "2.0.0"   # AttributeError: Read-only attribute
```

---

## __set_name__（Python 3.6+）

自动获取描述符在类中的名称：

```python
class Typed:
    def __set_name__(self, owner, name):
        """
        在类创建时自动调用
        owner: 拥有描述符的类
        name: 描述符在类中的属性名
        """
        self.name = name
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)

class Person:
    # 不需要传递 'name' 和 'age' 参数了
    name = Typed()
    age = Typed()

    def __init__(self, name, age):
        self.name = name
        self.age = age
```

---

## 内置描述符

### property 是描述符

```python
# property 的简化实现
class property:
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)
```

### classmethod 和 staticmethod

```python
# classmethod 的简化实现
class classmethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if objtype is None:
            objtype = type(obj)

        def bound_method(*args, **kwargs):
            return self.func(objtype, *args, **kwargs)

        return bound_method

# staticmethod 的简化实现
class staticmethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        return self.func
```

### 方法也是描述符

```python
class MyClass:
    def method(self):
        pass

# 函数对象有 __get__ 方法
print(type(MyClass.method))  # <class 'function'>
print(type(MyClass().method))  # <class 'method'>

# 通过描述符协议绑定 self
```

---

## 描述符 vs property

| 特性 | property | 描述符 |
|------|----------|--------|
| 定义位置 | 类内部 | 独立类 |
| 复用性 | 低 | 高 |
| 复杂度 | 简单 | 中等 |
| 适用场景 | 单一属性 | 多属性复用 |

```python
# property: 简单场景
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value

# 描述符: 需要复用
class PositiveNumber:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if value < 0:
            raise ValueError(f"{self.name} must be positive")
        obj.__dict__[self.name] = value

class Circle:
    radius = PositiveNumber()

class Rectangle:
    width = PositiveNumber()
    height = PositiveNumber()
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 存储在描述符实例 | 所有对象共享 | 存储在 obj.__dict__ |
| 忘记处理 obj=None | 类访问出错 | 检查并返回 self |
| 非数据描述符被覆盖 | 实例属性优先 | 使用数据描述符 |
| 命名冲突 | 私有属性被覆盖 | 使用 __set_name__ |

---

## 小结

1. **描述符协议**：`__get__`、`__set__`、`__delete__`
2. **数据描述符**：优先级高于实例属性
3. **非数据描述符**：可被实例属性覆盖
4. **__set_name__**：自动获取属性名
5. **应用场景**：类型验证、惰性计算、只读属性

