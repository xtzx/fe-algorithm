# 元类 Metaclass

> 类的类，控制类的创建过程

## 什么是元类

在 Python 中，类也是对象。元类就是创建类的"类"。

```python
# 一切皆对象
class MyClass:
    pass

# MyClass 是 type 的实例
print(type(MyClass))  # <class 'type'>

# type 是所有类的默认元类
print(type(int))      # <class 'type'>
print(type(str))      # <class 'type'>
print(type(type))     # <class 'type'>（type 也是自己的实例）
```

```
实例关系:
                type (元类)
                  │
    ┌─────────────┼─────────────┐
    │             │             │
   int          str         MyClass (类)
    │             │             │
    1           "hi"         obj (实例)
```

---

## type() 动态创建类

```python
# 常规方式定义类
class MyClass:
    x = 10
    def method(self):
        return self.x

# 使用 type() 动态创建等价的类
MyClass = type(
    'MyClass',           # 类名
    (),                  # 基类元组
    {                    # 属性和方法字典
        'x': 10,
        'method': lambda self: self.x,
    }
)

# 两种方式创建的类功能相同
obj = MyClass()
print(obj.method())  # 10
```

### type() 签名

```python
type(name, bases, dict)
# name: str - 类名
# bases: tuple - 基类元组
# dict: dict - 类属性和方法
```

---

## 自定义元类

### 基础语法

```python
class MyMeta(type):
    """自定义元类"""

    def __new__(mcs, name, bases, namespace):
        """
        创建类对象
        mcs: 元类本身
        name: 类名
        bases: 基类元组
        namespace: 类的命名空间（属性和方法）
        """
        print(f"Creating class: {name}")
        return super().__new__(mcs, name, bases, namespace)

# 使用元类
class MyClass(metaclass=MyMeta):
    pass
# 输出: Creating class: MyClass
```

### __new__ vs __init__

```python
class MyMeta(type):
    def __new__(mcs, name, bases, namespace):
        """创建类对象（在类创建之前）"""
        print(f"__new__: Creating {name}")
        # 可以修改 namespace
        namespace['created_by'] = 'MyMeta'
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        """初始化类对象（在类创建之后）"""
        print(f"__init__: Initializing {name}")
        super().__init__(name, bases, namespace)

class MyClass(metaclass=MyMeta):
    pass
# 输出:
# __new__: Creating MyClass
# __init__: Initializing MyClass

print(MyClass.created_by)  # 'MyMeta'
```

### __call__ 控制实例化

```python
class SingletonMeta(type):
    """单例模式元类"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        控制类的实例化过程
        当调用 MyClass() 时触发
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Database initialized")

db1 = Database()  # Database initialized
db2 = Database()  # 不打印，返回同一实例
print(db1 is db2)  # True
```

---

## 实用元类示例

### 自动注册类

```python
class PluginMeta(type):
    """自动注册插件的元类"""
    registry = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # 不注册基类
        if bases:
            plugin_name = namespace.get('name', name.lower())
            mcs.registry[plugin_name] = cls

        return cls

class Plugin(metaclass=PluginMeta):
    """插件基类"""
    pass

class JSONPlugin(Plugin):
    name = 'json'
    def parse(self, data):
        return json.loads(data)

class XMLPlugin(Plugin):
    name = 'xml'
    def parse(self, data):
        return xml.parse(data)

# 自动注册
print(PluginMeta.registry)
# {'json': <class 'JSONPlugin'>, 'xml': <class 'XMLPlugin'>}

# 使用
def get_plugin(name):
    return PluginMeta.registry.get(name)

parser = get_plugin('json')()
```

### 强制实现方法

```python
class InterfaceMeta(type):
    """强制子类实现特定方法"""
    required_methods = []

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # 检查是否实现了必需方法（跳过基类）
        if bases:
            for method in mcs.required_methods:
                if method not in namespace:
                    raise TypeError(
                        f"Class {name} must implement method: {method}"
                    )

        return cls

class SerializerMeta(InterfaceMeta):
    required_methods = ['serialize', 'deserialize']

class Serializer(metaclass=SerializerMeta):
    pass

class JSONSerializer(Serializer):
    def serialize(self, data):
        return json.dumps(data)

    def deserialize(self, data):
        return json.loads(data)

# 这会报错
class BadSerializer(Serializer):
    def serialize(self, data):
        pass
    # TypeError: Class BadSerializer must implement method: deserialize
```

### 属性验证

```python
class ValidatedMeta(type):
    """自动添加属性验证"""

    def __new__(mcs, name, bases, namespace):
        # 查找所有 Typed 描述符并设置名称
        for key, value in namespace.items():
            if hasattr(value, '__set_name__'):
                value.__set_name__(None, key)

        return super().__new__(mcs, name, bases, namespace)

class Typed:
    expected_type = object

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be {self.expected_type.__name__}")
        obj.__dict__[self.name] = value

class String(Typed):
    expected_type = str

class Integer(Typed):
    expected_type = int

class Model(metaclass=ValidatedMeta):
    pass

class User(Model):
    name = String()
    age = Integer()

    def __init__(self, name, age):
        self.name = name
        self.age = age

user = User("Alice", 30)
user.age = "thirty"  # TypeError: age must be int
```

### ORM 风格模型

```python
class Field:
    def __init__(self, column_type):
        self.column_type = column_type
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

class ModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        # 收集所有字段
        fields = {}
        for key, value in namespace.items():
            if isinstance(value, Field):
                fields[key] = value

        namespace['_fields'] = fields
        namespace['_table_name'] = name.lower()

        return super().__new__(mcs, name, bases, namespace)

class Model(metaclass=ModelMeta):
    def save(self):
        fields = ', '.join(self._fields.keys())
        values = ', '.join(
            repr(getattr(self, f)) for f in self._fields
        )
        print(f"INSERT INTO {self._table_name} ({fields}) VALUES ({values})")

class User(Model):
    name = Field('VARCHAR(100)')
    email = Field('VARCHAR(200)')
    age = Field('INTEGER')

    def __init__(self, name, email, age):
        self.name = name
        self.email = email
        self.age = age

user = User("Alice", "alice@example.com", 30)
user.save()
# INSERT INTO user (name, email, age) VALUES ('Alice', 'alice@example.com', 30)
```

---

## __init_subclass__（Python 3.6+ 替代方案）

更简单的方式实现部分元类功能：

```python
class Plugin:
    registry = {}

    def __init_subclass__(cls, name=None, **kwargs):
        """子类被定义时自动调用"""
        super().__init_subclass__(**kwargs)
        plugin_name = name or cls.__name__.lower()
        Plugin.registry[plugin_name] = cls

class JSONPlugin(Plugin, name='json'):
    pass

class XMLPlugin(Plugin, name='xml'):
    pass

print(Plugin.registry)
# {'json': <class 'JSONPlugin'>, 'xml': <class 'XMLPlugin'>}
```

### 何时用 __init_subclass__ vs 元类

| 场景 | 推荐 |
|------|------|
| 简单注册 | `__init_subclass__` |
| 验证子类 | `__init_subclass__` |
| 修改类创建过程 | 元类 |
| 控制实例化 | 元类 |
| 多重继承需求 | `__init_subclass__` |

---

## 元类继承

```python
class MetaA(type):
    pass

class MetaB(type):
    pass

class A(metaclass=MetaA):
    pass

class B(metaclass=MetaB):
    pass

# 错误：元类冲突
# class C(A, B):  # TypeError: metaclass conflict
#     pass

# 解决：创建共同元类
class MetaC(MetaA, MetaB):
    pass

class C(A, B, metaclass=MetaC):
    pass
```

---

## 慎用元类

### 元类的问题

1. **复杂性**：难以理解和调试
2. **继承冲突**：多元类继承复杂
3. **可读性**：隐式行为难以追踪

### 替代方案

```python
# 1. 装饰器
def register(cls):
    registry[cls.__name__] = cls
    return cls

@register
class MyPlugin:
    pass

# 2. __init_subclass__
class Base:
    def __init_subclass__(cls, **kwargs):
        # 处理子类
        pass

# 3. 描述符
class Validated:
    def __set_name__(self, owner, name):
        # 处理属性
        pass
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 元类冲突 | 多继承时元类不兼容 | 创建共同元类 |
| 过度使用 | 增加复杂性 | 优先用装饰器或 __init_subclass__ |
| __new__ vs __init__ | 混淆时机 | __new__ 创建，__init__ 初始化 |
| 忘记 super() | 打断继承链 | 始终调用 super() |

---

## 小结

1. **元类是类的类**：控制类的创建
2. **type()**：动态创建类
3. **__new__**：创建类对象
4. **__call__**：控制实例化
5. **替代方案**：`__init_subclass__`、装饰器、描述符
6. **使用原则**：除非必要，避免使用元类

