# 动态属性

> __getattr__、__setattr__、__getattribute__ 与动态类创建

## 属性访问魔术方法

### __getattr__

当属性**不存在**时调用：

```python
class DynamicObject:
    def __init__(self):
        self.existing = "I exist"

    def __getattr__(self, name):
        """属性不存在时调用"""
        print(f"__getattr__ called for: {name}")
        return f"dynamic_{name}"

obj = DynamicObject()
print(obj.existing)      # "I exist"（不调用 __getattr__）
print(obj.missing)       # __getattr__ called for: missing → "dynamic_missing"
```

### __getattribute__

**所有**属性访问都会调用：

```python
class LoggedObject:
    def __init__(self):
        self.value = 42

    def __getattribute__(self, name):
        """所有属性访问都会调用"""
        print(f"Accessing: {name}")
        # 必须用 object.__getattribute__ 避免递归
        return object.__getattribute__(self, name)

obj = LoggedObject()
print(obj.value)  # Accessing: value → 42
```

### __setattr__

设置属性时调用：

```python
class ValidatedObject:
    def __setattr__(self, name, value):
        """设置任何属性时调用"""
        print(f"Setting {name} = {value}")
        # 必须用 object.__setattr__ 避免递归
        object.__setattr__(self, name, value)

obj = ValidatedObject()
obj.x = 10  # Setting x = 10
```

### __delattr__

删除属性时调用：

```python
class TrackedObject:
    def __delattr__(self, name):
        print(f"Deleting: {name}")
        object.__delattr__(self, name)

obj = TrackedObject()
obj.x = 10
del obj.x  # Deleting: x
```

---

## 属性访问顺序

```
obj.attr 的查找顺序:

1. __getattribute__ 被调用
2. 数据描述符（__get__ + __set__）
3. obj.__dict__['attr']
4. type(obj).__dict__['attr']（及其基类）
5. 非数据描述符（只有 __get__）
6. __getattr__（如果定义了且以上都没找到）
7. AttributeError
```

```python
class Demo:
    class_attr = "class"

    def __init__(self):
        self.__dict__['instance_attr'] = "instance"

    def __getattribute__(self, name):
        print(f"1. __getattribute__: {name}")
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        print(f"6. __getattr__: {name}")
        return f"dynamic_{name}"

obj = Demo()
print(obj.class_attr)     # 1 → "class"
print(obj.instance_attr)  # 1 → "instance"
print(obj.missing)        # 1 → 6 → "dynamic_missing"
```

---

## 实用示例

### 代理对象

```python
class Proxy:
    """代理另一个对象的所有属性访问"""

    def __init__(self, target):
        object.__setattr__(self, '_target', target)

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._target, name, value)

class RealObject:
    def __init__(self):
        self.value = 42

    def method(self):
        return self.value * 2

real = RealObject()
proxy = Proxy(real)

print(proxy.value)      # 42
print(proxy.method())   # 84
proxy.value = 100
print(real.value)       # 100（修改了原对象）
```

### 链式 API

```python
class QueryBuilder:
    """链式查询构建器"""

    def __init__(self):
        self._parts = []

    def __getattr__(self, name):
        self._parts.append(name)
        return self

    def __call__(self, *args, **kwargs):
        self._parts.append(f"({args}, {kwargs})")
        return self

    def __str__(self):
        return ".".join(self._parts)

q = QueryBuilder()
print(q.users.filter(active=True).order_by('created_at'))
# users.filter((()), {'active': True}).order_by((('created_at',), {}))
```

### 配置对象

```python
class Config:
    """支持点号访问的配置对象"""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def __getattr__(self, name):
        raise AttributeError(f"Config has no attribute: {name}")

    def __repr__(self):
        return f"Config({self.__dict__})"

config = Config({
    'database': {
        'host': 'localhost',
        'port': 5432,
    },
    'debug': True,
})

print(config.debug)           # True
print(config.database.host)   # localhost
print(config.database.port)   # 5432
```

### 惰性属性

```python
class LazyProperty:
    """只在首次访问时计算的属性"""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.name, value)  # 替换为计算结果
        return value

class ExpensiveComputation:
    def __init__(self, data):
        self.data = data

    @LazyProperty
    def result(self):
        print("Computing...")
        return sum(x ** 2 for x in self.data)

obj = ExpensiveComputation(range(1000000))
print(obj.result)  # Computing... 结果
print(obj.result)  # 直接返回缓存的结果（不计算）
```

---

## 动态创建类

### 使用 type()

```python
# 动态创建类
def make_class(name, attributes):
    return type(name, (), attributes)

Person = make_class('Person', {
    'greet': lambda self: f"Hello, I'm {self.name}",
    '__init__': lambda self, name: setattr(self, 'name', name),
})

p = Person("Alice")
print(p.greet())  # Hello, I'm Alice
```

### 动态添加方法

```python
class MyClass:
    pass

# 动态添加方法
def dynamic_method(self):
    return "I'm dynamic!"

MyClass.dynamic_method = dynamic_method

# 使用 types.MethodType 绑定到实例
import types

obj = MyClass()
obj.instance_method = types.MethodType(
    lambda self: "Instance bound!",
    obj
)

print(obj.dynamic_method())  # I'm dynamic!
print(obj.instance_method()) # Instance bound!
```

### 工厂函数

```python
def create_model(name: str, fields: dict):
    """动态创建数据模型类"""

    def __init__(self, **kwargs):
        for field in fields:
            setattr(self, field, kwargs.get(field))

    def __repr__(self):
        values = ', '.join(f"{k}={getattr(self, k)!r}" for k in fields)
        return f"{name}({values})"

    return type(name, (), {
        '__init__': __init__,
        '__repr__': __repr__,
        '_fields': fields,
    })

# 使用
User = create_model('User', {
    'name': str,
    'email': str,
    'age': int,
})

user = User(name="Alice", email="alice@example.com", age=30)
print(user)  # User(name='Alice', email='alice@example.com', age=30)
```

---

## __slots__ 与动态属性

```python
class SlottedClass:
    __slots__ = ['x', 'y']

obj = SlottedClass()
obj.x = 10
obj.y = 20
# obj.z = 30  # AttributeError: 'SlottedClass' object has no attribute 'z'

# __slots__ 禁用了 __dict__，提高内存效率
# 但也禁用了动态添加属性
```

### __slots__ + __dict__

```python
class HybridClass:
    __slots__ = ['x', 'y', '__dict__']  # 允许动态属性

obj = HybridClass()
obj.x = 10
obj.z = 30  # 现在可以了
```

---

## 注意事项

### 避免递归

```python
class BadExample:
    def __setattr__(self, name, value):
        # ❌ 无限递归！
        self.log = f"Setting {name}"
        self.__dict__[name] = value

class GoodExample:
    def __setattr__(self, name, value):
        # ✅ 使用 object.__setattr__
        object.__setattr__(self, name, value)
        # 或直接操作 __dict__
        # self.__dict__[name] = value
```

### __getattribute__ 的陷阱

```python
class Tricky:
    def __getattribute__(self, name):
        # ❌ 无限递归！
        print(f"Accessing {self.name}")  # self.name 再次触发 __getattribute__
        return object.__getattribute__(self, name)

    def __getattribute__(self, name):
        # ✅ 正确
        print(f"Accessing {name}")
        return object.__getattribute__(self, name)
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 递归调用 | __setattr__ 中用 self.x | 用 object.__setattr__ |
| __getattr__ vs __getattribute__ | 混淆调用时机 | getattr 仅不存在时调用 |
| __slots__ 限制 | 不能动态添加属性 | 加入 __dict__ 到 slots |
| 性能问题 | __getattribute__ 太慢 | 避免过度使用 |

---

## 小结

1. **__getattr__**：属性不存在时调用
2. **__getattribute__**：所有属性访问都调用
3. **__setattr__**：设置属性时调用
4. **动态类**：使用 `type()` 创建
5. **注意递归**：使用 `object.__xxx__` 避免

