# Python 完整学习提示词（共 26 个阶段）

> 面向：资深前端（JS/TS 背景），零 Python 基础
> 目标：从语法到工程化到 AI 应用，全栈掌握 Python
> 结构：21 个知识阶段 + 5 个综合项目阶段

---

## 📊 学习路线总览

```
模块 A：Python 语言基础（P01-P05）
├── P01: 基础语法（JS 对照）
├── P02: 容器与数据结构
├── P03: 面向对象编程
├── P04: 函数式与装饰器
└── P05: 标准库精选

模块 B：工程化体系（P06-P10）
├── P06: Python 运行时原理
├── P07: 包与环境管理
├── P08: 工程质量工具链
├── P09: 测试体系
└── P10: 调试与性能优化

🎯 综合项目 1（P11）：CLI 工具项目

模块 C：数据处理与自动化（P12-P14）
├── P12: 数据处理与模型化
├── P13: 文件自动化
└── P14: 工程化脚手架

🎯 综合项目 2（P15）：自动化脚本项目

模块 D：网络与并发（P16-P18）
├── P16: HTTP 客户端工程化
├── P17: asyncio 并发
└── P18: 爬虫工程化

🎯 综合项目 3（P19）：数据采集项目

模块 E：后端服务（P20-P22）
├── P20: FastAPI 服务
├── P21: 存储与缓存
└── P22: 部署与可观测性

🎯 综合项目 4（P23）：API 服务项目

模块 F：AI 工程（P24-P25）
├── P24: LLM 客户端与 RAG
└── P25: AI 服务安全与评测

🎯 终极项目（P26）：AI 知识库助手
```

---

## 📁 每阶段统一输出格式

每个阶段要求 AI 输出一个完整项目文件夹：
1. 先输出目录树
2. 再逐文件输出完整内容
3. 所有示例必须可运行
4. docs 里包含：概念、示例、常见坑、JS 对照（适用时）、面试问答

---

# 模块 A：Python 语言基础

---

## P01-python-basics.prompt.md

你现在是「Python 语法导师」，面向 JS/TS 资深工程师。
目标：用 JS 对照的方式，快速掌握 Python 基础语法。

【本次主题】P01：Python 基础语法

【前置要求】无（零基础友好）

【学完后能做】
- 读懂 Python 代码
- 写出基本的 Python 脚本
- 理解 Python 与 JS 的核心差异

【必须覆盖】

1) 开发环境：
   - Python 安装、版本管理（pyenv 概念）
   - REPL、python -c、python script.py
   - VS Code 配置（Python 插件、设置）

2) 变量与类型（JS 对照）：
   - 动态类型 vs JS 动态类型
   - 基本类型：int、float、str、bool、None
   - 类型转换：int()、str()、float()、bool()
   - type() 和 isinstance()

3) 运算符（JS 对照）：
   - 算术：+、-、*、/、//（整除）、%、**（幂）
   - 比较：==、!=、<、>、<=、>=、is（身份）
   - 逻辑：and、or、not（vs JS &&、||、!）
   - 成员：in、not in
   - 三元：x if condition else y（vs JS ? :）

4) 字符串（JS 对照）：
   - 引号：单引号、双引号、三引号（多行）
   - f-string：f"Hello {name}"（vs JS 模板字符串）
   - 常用方法：strip、split、join、replace、find、startswith
   - 索引与切片：s[0]、s[-1]、s[1:3]、s[::2]
   - 编码：encode/decode、UTF-8

5) 控制流（JS 对照）：
   - if/elif/else（vs JS if/else if/else）
   - for 循环：for x in iterable（vs JS for...of）
   - range()：range(10)、range(1, 10)、range(0, 10, 2)
   - while 循环
   - break、continue、else 子句（循环的 else）
   - match-case（Python 3.10+，vs JS switch）

6) 函数基础（JS 对照）：
   - def 定义（vs function/arrow function）
   - 参数：位置参数、默认参数、关键字参数
   - 返回值：return、多返回值（元组解包）
   - 文档字符串：docstring
   - 作用域：LEGB 规则、global、nonlocal

7) 输入输出：
   - print()：sep、end、file 参数
   - input()
   - 文件读写：open()、with 语句、read/write/readlines

【练习题】25 道（含答案与思路）
- 基础 10 道（类型、字符串、控制流）
- 进阶 10 道（函数、文件）
- 挑战 5 道（综合应用）

【面试高频】至少 10 个
- Python 2 vs Python 3 主要区别？
- is 和 == 的区别？
- 可变对象 vs 不可变对象？
- 为什么 Python 没有 switch（3.10 之前）？
- Python 的作用域规则？
- 如何交换两个变量的值？
- range 和 xrange 的区别（历史问题）？
- Python 字符串是可变的吗？
- f-string vs format() vs % 的区别？
- Python 的 True/False 与 JS 的 truthy/falsy 对比？

【常见坑】
- 缩进错误（Python 靠缩进而非大括号）
- 可变默认参数陷阱：def f(lst=[])
- 整数除法：3 / 2 = 1.5（Python 3）vs 1（Python 2）
- 字符串不可变：s[0] = 'a' 报错
- is vs ==：[] is [] 为 False

【JS 对照表】必须包含
- 控制流对照
- 运算符对照
- 函数定义对照
- 字符串操作对照

【小项目】
文本统计器：统计文件的行数、单词数、字符数、最长行

【输出形式】
/py-01-basics/
├── README.md
├── docs/
│   ├── 01-environment-setup.md
│   ├── 02-variables-and-types.md
│   ├── 03-operators.md
│   ├── 04-strings.md
│   ├── 05-control-flow.md
│   ├── 06-functions.md
│   ├── 07-file-io.md
│   ├── 08-js-comparison-table.md
│   ├── 09-exercises.md
│   └── 10-interview-questions.md
├── examples/
│   ├── 01_hello.py
│   ├── 02_types_demo.py
│   ├── 03_strings_demo.py
│   ├── 04_control_flow.py
│   ├── 05_functions.py
│   └── 06_file_io.py
├── exercises/
│   ├── basic/
│   ├── advanced/
│   └── challenge/
├── project/
│   └── text_analyzer/
│       ├── main.py
│       └── test_analyzer.py
└── scripts/
    └── run_all.sh

要求：
- 所有示例在 Python 3.12+ 可运行
- README 必须包含：快速开始、与 JS 的核心差异总结
- 输出顺序：先目录树，再逐文件完整内容

---

## P02-containers.prompt.md

你现在是「Python 数据结构导师」，面向 JS/TS 资深工程师。
目标：掌握 Python 核心容器类型，对比 JS Array/Object/Set/Map。

【本次主题】P02：容器与数据结构

【前置要求】完成 P01

【学完后能做】
- 熟练使用 list/tuple/dict/set
- 写出 Pythonic 的推导式代码
- 理解可变/不可变、可哈希的概念

【必须覆盖】

1) 列表 list（vs JS Array）：
   - 创建：[]、list()、list(range())
   - 操作：append、extend、insert、pop、remove、clear
   - 切片：l[1:3]、l[::2]、l[::-1]（反转）
   - 查找：index、count、in
   - 排序：sort()（原地）、sorted()（新列表）、key 参数
   - 复制：浅拷贝 vs 深拷贝（copy、deepcopy）

2) 元组 tuple（JS 无直接对应）：
   - 创建：()、tuple()、(1,)（单元素）
   - 不可变性：为什么用元组
   - 解包：a, b = (1, 2)、*rest 解包
   - 命名元组：namedtuple、typing.NamedTuple

3) 字典 dict（vs JS Object/Map）：
   - 创建：{}、dict()、dict(a=1)
   - 操作：get、setdefault、update、pop、keys/values/items
   - 遍历：for k, v in d.items()
   - 推导式：{k: v for k, v in ...}
   - 有序性：Python 3.7+ dict 保序
   - defaultdict、Counter

4) 集合 set（vs JS Set）：
   - 创建：set()、{1, 2, 3}（注意 {} 是空 dict）
   - 操作：add、remove、discard、pop、clear
   - 集合运算：union、intersection、difference、symmetric_difference
   - frozenset：不可变集合

5) 推导式（Comprehensions）：
   - 列表推导式：[x*2 for x in range(10)]
   - 条件过滤：[x for x in range(10) if x % 2 == 0]
   - 嵌套推导式：[[i*j for j in range(3)] for i in range(3)]
   - 字典推导式：{k: v for k, v in pairs}
   - 集合推导式：{x*2 for x in range(10)}
   - 生成器表达式：(x*2 for x in range(10))

6) 序列操作：
   - 通用操作：len、min、max、sum、sorted、reversed
   - 索引与切片的高级用法
   - zip、enumerate、map、filter
   - any、all

7) 可变 vs 不可变：
   - 可变：list、dict、set
   - 不可变：int、str、tuple、frozenset
   - 作为函数参数的行为差异
   - 作为 dict 键的要求（可哈希）

【练习题】25 道
- list 操作 8 道
- dict 操作 8 道
- 推导式 6 道
- 综合 3 道

【面试高频】至少 10 个
- list 和 tuple 的区别？什么时候用 tuple？
- dict 的键必须满足什么条件？
- 如何合并两个字典？
- 推导式和 map/filter 哪个更 Pythonic？
- 浅拷贝和深拷贝的区别？
- 如何去重并保持顺序？
- 如何找出两个列表的交集？
- dict.get() 和直接 [] 访问的区别？
- Python 的 dict 是如何实现的（哈希表）？
- 为什么字符串可以作为 dict 的键，列表不行？

【常见坑】
- 空集合必须用 set()，{} 是空字典
- 列表作为默认参数的陷阱
- 浅拷贝：嵌套对象仍是引用
- 遍历时修改列表
- dict.keys() 返回的是视图不是列表

【小项目】
词频统计器：读取文本，统计每个单词出现次数，输出 Top N

【输出形式】
/py-02-containers/
├── README.md
├── docs/
│   ├── 01-list.md
│   ├── 02-tuple.md
│   ├── 03-dict.md
│   ├── 04-set.md
│   ├── 05-comprehensions.md
│   ├── 06-sequence-operations.md
│   ├── 07-mutable-immutable.md
│   ├── 08-js-comparison.md
│   ├── 09-exercises.md
│   └── 10-interview-questions.md
├── examples/
│   ├── 01_list_demo.py
│   ├── 02_tuple_demo.py
│   ├── 03_dict_demo.py
│   ├── 04_set_demo.py
│   ├── 05_comprehensions.py
│   └── 06_copy_demo.py
├── exercises/
├── project/
│   └── word_frequency/
└── scripts/

---

## P03-oop.prompt.md

你现在是「Python 面向对象编程导师」，面向 JS/TS 资深工程师。
目标：掌握 Python OOP，对比 JS class/prototype。

【本次主题】P03：面向对象编程

【前置要求】完成 P01-P02

【学完后能做】
- 设计和实现 Python 类
- 理解魔法方法和协议
- 使用 dataclass 简化数据类

【必须覆盖】

1) 类与实例（vs JS class）：
   - class 定义、__init__ 构造器
   - self 参数（vs JS this）
   - 实例属性 vs 类属性
   - 实例方法、类方法（@classmethod）、静态方法（@staticmethod）

2) 继承与多态：
   - 单继承：class Child(Parent)
   - super() 调用父类方法
   - 方法重写（Override）
   - 多继承与 MRO（方法解析顺序）
   - isinstance() 和 issubclass()

3) 魔法方法（Dunder Methods）：
   - 构造/析构：__init__、__new__、__del__
   - 字符串表示：__str__、__repr__
   - 比较：__eq__、__ne__、__lt__、__le__、__gt__、__ge__
   - 哈希：__hash__
   - 布尔：__bool__
   - 容器协议：__len__、__getitem__、__setitem__、__delitem__、__contains__、__iter__
   - 可调用：__call__
   - 上下文管理：__enter__、__exit__

4) 属性与描述符：
   - @property：getter/setter/deleter
   - 属性验证
   - 描述符协议（__get__、__set__、__delete__）概念

5) 抽象类与协议：
   - abc.ABC 和 @abstractmethod
   - typing.Protocol（结构化子类型）
   - 鸭子类型 vs 显式接口

6) dataclass（Python 3.7+）：
   - @dataclass 装饰器
   - field() 配置
   - frozen=True 不可变
   - 与 pydantic 的对比（预告 P12）
   - 与 TypedDict 的对比

7) 特殊模式：
   - 单例模式
   - 工厂模式
   - Mixin 类

【练习题】20 道
- 类基础 6 道
- 魔法方法 6 道
- 继承 4 道
- dataclass 4 道

【面试高频】至少 10 个
- Python 中如何实现私有属性？（单下划线、双下划线）
- __new__ 和 __init__ 的区别？
- 类方法和静态方法的区别？什么时候用？
- MRO 是什么？如何查看？
- __str__ 和 __repr__ 的区别？
- 如何让自定义对象可以用 len()？
- 如何让自定义对象可以用 for 遍历？
- dataclass 和普通 class 的区别？
- Python 的多继承有什么问题？如何解决？
- 描述符是什么？property 的底层原理？

【常见坑】
- 可变类属性被所有实例共享
- 忘记调用 super().__init__()
- 双下划线名称改写（name mangling）
- __eq__ 实现后 __hash__ 被设为 None
- 钻石继承问题

【小项目】
扑克牌游戏：实现 Card、Deck 类，支持洗牌、发牌、排序

【输出形式】
/py-03-oop/
├── README.md
├── docs/
│   ├── 01-class-basics.md
│   ├── 02-inheritance.md
│   ├── 03-magic-methods.md
│   ├── 04-properties.md
│   ├── 05-abstract-protocol.md
│   ├── 06-dataclass.md
│   ├── 07-design-patterns.md
│   ├── 08-js-comparison.md
│   ├── 09-exercises.md
│   └── 10-interview-questions.md
├── examples/
├── exercises/
├── project/
│   └── poker_game/
└── scripts/

---

## P04-functional.prompt.md

你现在是「Python 函数式编程导师」，面向 JS/TS 资深工程师。
目标：掌握装饰器、闭包、生成器等高级函数特性。

【本次主题】P04：函数式与装饰器

【前置要求】完成 P01-P03

【学完后能做】
- 编写和理解装饰器
- 使用生成器处理大数据
- 运用函数式编程思想

【必须覆盖】

1) 高阶函数：
   - 函数作为参数
   - 函数作为返回值
   - map、filter、reduce（functools.reduce）
   - sorted 的 key 参数
   - 与 JS 高阶函数对比

2) lambda 表达式：
   - 语法：lambda x: x * 2
   - 使用场景：排序 key、简单回调
   - 限制：只能是单表达式
   - vs JS 箭头函数

3) 闭包（Closure）：
   - 定义与原理
   - 状态保持
   - nonlocal 关键字
   - 常见用途：工厂函数、延迟计算
   - vs JS 闭包

4) 装饰器（Decorator）：
   - 基础装饰器：@decorator
   - 带参数的装饰器
   - 多个装饰器叠加（执行顺序）
   - functools.wraps 保留元信息
   - 类装饰器
   - 装饰类的装饰器
   - 常用装饰器：@property、@classmethod、@staticmethod、@dataclass
   - 实战装饰器：计时器、重试、缓存、权限检查

5) 生成器（Generator）：
   - yield 关键字
   - 生成器函数 vs 普通函数
   - 生成器表达式：(x for x in range(10))
   - 惰性求值的优势
   - 大数据处理场景
   - yield from（委托生成器）
   - 生成器的 send() 和 close()

6) 迭代器协议：
   - __iter__ 和 __next__
   - iter() 和 next()
   - StopIteration
   - 自定义迭代器类
   - itertools 模块入门

7) functools 模块：
   - partial：偏函数
   - lru_cache：缓存装饰器
   - reduce：折叠操作
   - wraps：装饰器辅助
   - total_ordering：比较方法补全

8) itertools 模块：
   - count、cycle、repeat
   - chain、zip_longest
   - groupby
   - combinations、permutations
   - takewhile、dropwhile

【练习题】25 道
- 高阶函数 5 道
- 装饰器 8 道
- 生成器 7 道
- itertools 5 道

【面试高频】至少 10 个
- 什么是装饰器？手写一个计时装饰器？
- 装饰器的执行顺序是怎样的？
- 生成器和列表的区别？什么时候用生成器？
- yield 和 return 的区别？
- lru_cache 的原理和使用场景？
- 如何实现一个带参数的装饰器？
- 闭包是什么？Python 中如何创建闭包？
- itertools.groupby 的使用注意事项？
- 生成器如何处理大文件？
- 什么是惰性求值？

【常见坑】
- 装饰器丢失函数元信息（需要 @wraps）
- 闭包变量绑定问题（循环中的 lambda）
- 生成器只能迭代一次
- groupby 要求数据已排序
- lru_cache 用于可变参数会出错

【小项目】
实用装饰器库：实现 @timer、@retry、@cache、@validate 装饰器

【输出形式】
/py-04-functional/
├── README.md
├── docs/
│   ├── 01-higher-order-functions.md
│   ├── 02-lambda.md
│   ├── 03-closure.md
│   ├── 04-decorators.md
│   ├── 05-generators.md
│   ├── 06-iterators.md
│   ├── 07-functools.md
│   ├── 08-itertools.md
│   ├── 09-exercises.md
│   └── 10-interview-questions.md
├── examples/
├── exercises/
├── project/
│   └── decorator_lib/
└── scripts/

---

## P05-stdlib.prompt.md

你现在是「Python 标准库导师」，面向 JS/TS 资深工程师。
目标：掌握最常用的标准库模块。

【本次主题】P05：标准库精选

【前置要求】完成 P01-P04

【学完后能做】
- 熟练使用常见标准库
- 不依赖第三方库完成常见任务
- 理解标准库设计思想

【必须覆盖】

1) pathlib（现代文件路径，vs Node path）：
   - Path 对象创建
   - 路径拼接：/运算符
   - 路径属性：name、stem、suffix、parent、parts
   - 路径操作：exists、is_file、is_dir、mkdir、rmdir、unlink
   - 文件操作：read_text、write_text、read_bytes
   - 遍历：iterdir、glob、rglob
   - 与 os.path 对比

2) collections（高级容器）：
   - Counter：计数器
   - defaultdict：默认值字典
   - OrderedDict：有序字典（Python 3.7+ dict 已有序）
   - deque：双端队列
   - namedtuple：命名元组
   - ChainMap：字典链

3) datetime（日期时间，vs JS Date）：
   - date、time、datetime、timedelta
   - 创建：now()、today()、fromisoformat()
   - 格式化：strftime、strptime
   - 时区：timezone、zoneinfo（Python 3.9+）
   - 常见操作：日期加减、比较

4) re（正则表达式，vs JS RegExp）：
   - 匹配：match、search、fullmatch
   - 查找：findall、finditer
   - 替换：sub、subn
   - 分割：split
   - 编译：compile
   - 分组：groups、groupdict
   - 常用模式：邮箱、URL、电话

5) json（vs JS JSON）：
   - dumps、loads
   - dump、load（文件）
   - 自定义编码器/解码器
   - ensure_ascii、indent、default 参数

6) os 与 shutil：
   - 环境变量：os.environ、getenv
   - 进程：os.system、subprocess（概念）
   - 文件操作：shutil.copy、move、rmtree
   - 临时文件：tempfile

7) logging（日志）：
   - 日志级别：DEBUG、INFO、WARNING、ERROR、CRITICAL
   - 基本配置：basicConfig
   - Logger、Handler、Formatter
   - 最佳实践

8) argparse（命令行参数）：
   - ArgumentParser
   - add_argument：位置参数、可选参数、类型、默认值、帮助
   - 子命令：add_subparsers
   - vs click（预告）

9) typing（类型提示）：
   - 基础类型：int、str、bool、None
   - 容器类型：List、Dict、Set、Tuple
   - Optional、Union、Any
   - Callable、TypeVar、Generic
   - Literal、Final
   - TypedDict、Protocol

10) 其他常用模块概览：
    - random：随机数
    - uuid：唯一标识
    - hashlib：哈希
    - base64：编解码
    - urllib：URL 处理
    - secrets：安全随机

【练习题】25 道
- pathlib 5 道
- collections 5 道
- datetime 5 道
- re 5 道
- 综合 5 道

【面试高频】至少 10 个
- pathlib 和 os.path 的区别？
- Counter 最常见的用法？
- 如何处理 Python 的时区问题？
- 正则表达式的贪婪和非贪婪？
- logging 的最佳实践是什么？
- 如何解析命令行参数？
- typing 模块中 Optional 和 Union 的区别？
- 如何生成安全的随机数？
- json.dumps 处理自定义对象？
- 如何用 Python 处理环境变量？

【小项目】
文件整理器：按扩展名分类文件、重命名、生成报告

【输出形式】
/py-05-stdlib/
├── README.md
├── docs/
│   ├── 01-pathlib.md
│   ├── 02-collections.md
│   ├── 03-datetime.md
│   ├── 04-re.md
│   ├── 05-json.md
│   ├── 06-os-shutil.md
│   ├── 07-logging.md
│   ├── 08-argparse.md
│   ├── 09-typing.md
│   ├── 10-other-modules.md
│   ├── 11-exercises.md
│   └── 12-interview-questions.md
├── examples/
├── exercises/
├── project/
│   └── file_organizer/
└── scripts/

---

# 模块 B：工程化体系

---

## P06-runtime.prompt.md

你现在是「Python 运行时与解释器原理导师」。
面向：有 JS/TS 背景、熟悉 V8/JIT 概念，想深入理解 Python 工作原理。

【本次主题】P06：Python 运行时原理

【前置要求】完成 P01-P05

【学完后能做】
- 理解 Python 代码如何执行
- 正确选择 async/线程/进程
- 诊断性能和内存问题

【必须覆盖】

1) CPython 执行链路：
   - 源码 → AST → 字节码 → 虚拟机执行
   - compile() 和 ast 模块演示
   - dis 反汇编
   - .pyc 与 __pycache__：何时生成、何时失效

2) 与 JS/V8 对照：
   - 解释器/字节码 vs JIT
   - PyPy/Numba 等"JIT路线"概览

3) GIL 与并发：
   - 什么是 GIL
   - 线程为何不适合 CPU 密集
   - I/O 密集为何仍有效
   - threading vs multiprocessing vs asyncio 选择表

4) import 系统：
   - sys.path 查找顺序
   - 包/模块区别
   - 命名空间包
   - python -m 运行方式
   - __main__.py 的作用
   - 相对导入 vs 绝对导入

5) 内存与对象模型：
   - 引用计数机制
   - GC 处理循环引用
   - gc 模块演示
   - tracemalloc 入门
   - 对象的 id() 和 is

6) 错误与 traceback：
   - raise from 异常链
   - 如何读懂 traceback
   - 异常层级结构

7) 性能工具入门：
   - cProfile 使用
   - tracemalloc 内存追踪
   - time.perf_counter 计时

【练习题】15 道

【面试高频】至少 10 个
- 什么是 GIL？它对多线程有什么影响？
- .pyc 文件是什么？什么时候重新生成？
- Python 的内存管理机制是什么？
- 如何定位 Python 程序的内存泄漏？
- async 的本质是什么？
- 什么时候用多进程？什么时候用多线程？
- import 时 Python 如何查找模块？
- __name__ == "__main__" 的作用？
- 如何优化 Python 程序性能？
- PyPy 和 CPython 的区别？

【输出形式】
/py-06-runtime/
├── README.md
├── docs/
│   ├── 01-cpython-execution.md
│   ├── 02-bytecode-pyc.md
│   ├── 03-gil-concurrency.md
│   ├── 04-import-system.md
│   ├── 05-memory-gc.md
│   ├── 06-traceback.md
│   ├── 07-performance-tools.md
│   └── 08-interview-questions.md
├── examples/
└── scripts/

---

## P07-packaging.prompt.md

你现在是「Python 包管理与环境专家」。
目标：像管理 Node 依赖一样管理 Python：可重复、可锁定、可迁移。

【本次主题】P07：包与环境管理

【前置要求】完成 P06

【学完后能做】
- 正确配置 pyproject.toml
- 使用 uv/poetry 管理依赖
- 搭建可重复的开发环境

【必须覆盖】

1) pyproject.toml：
   - 项目元数据
   - 依赖声明
   - PEP 517/518/621
   - 与 package.json 对比

2) venv 本质：
   - 解释器隔离
   - site-packages
   - PATH 配置

3) 包格式：
   - wheel vs sdist
   - 构建过程
   - build backend：setuptools/hatchling/poetry-core

4) 依赖管理工具对比：
   - uv（优先推荐）：依赖解析、lock、同步
   - poetry：lock 与 workspace
   - pdm：PEP 582 思路
   - pip-tools：requirements.in → requirements.txt

5) lockfile 最佳实践：
   - 为什么需要 lock
   - 如何处理私有包
   - 最小化依赖漂移

6) entry points：
   - project.scripts
   - console_scripts

7) 私有源/镜像：
   - pip config
   - index-url、extra-index-url

8) 多环境管理：
   - dev/test/prod 依赖分组
   - optional dependencies

9) CI 最佳实践：
   - 缓存策略
   - 可重复构建

【练习题】10 道

【面试高频】至少 10 个
- wheel 和 sdist 的区别？
- pyproject.toml 的作用？
- 什么是 editable install？
- 如何处理依赖冲突？
- lockfile 的作用是什么？
- uv 和 pip 的区别？
- 如何创建一个 Python 包？
- 如何发布到 PyPI？
- PEP 517 是什么？
- 如何管理多 Python 版本？

【输出形式】
/py-07-packaging/
├── README.md
├── pyproject.toml
├── docs/
├── src/
│   └── packaging_lab/
├── tests/
└── scripts/

---

## P08-quality.prompt.md

你现在是「Python 工程质量工具链导师」。
目标：给出等价于 ESLint/Prettier/TS 的 Python 方案。

【本次主题】P08：工程质量工具链

【前置要求】完成 P07

【学完后能做】
- 配置 ruff/black/pyright
- 使用 pre-commit 自动化检查
- 保持代码质量

【必须覆盖】

1) ruff：
   - lint 规则
   - import 排序
   - format（可选）
   - 配置方式

2) black/ruff format：
   - 格式化策略
   - 与团队协作

3) 类型检查：
   - pyright（优先）vs mypy
   - 配置与使用
   - strict 模式

4) pre-commit：
   - 等价于 husky + lint-staged
   - 配置文件
   - 常用钩子

5) 任务编排：
   - make
   - just
   - nox/tox

【练习题】8 道

【面试高频】至少 8 个
- ruff 和 black 的区别？
- pyright 和 mypy 的区别？
- pre-commit 的原理？
- 如何在 CI 中集成代码检查？
- Python 的类型检查是强制的吗？
- 如何处理第三方库没有类型的问题？
- py.typed 是什么？
- 如何忽略某行的类型检查？

【输出形式】
/py-08-quality/
├── README.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── docs/
├── src/
│   └── quality_demo/
│       ├── sample_good.py
│       └── sample_bad.py（故意放 6+ 个问题）
└── scripts/

---

## P09-testing.prompt.md

你现在是「Python 测试体系导师」。
目标：建立 pytest 思维、mock 技巧、覆盖率意识。

【本次主题】P09：测试体系

【前置要求】完成 P08

【学完后能做】
- 编写 pytest 测试
- 合理使用 mock
- 理解测试金字塔

【必须覆盖】

1) pytest 基础：
   - 测试发现与命名
   - assert 语句
   - 参数化：@pytest.mark.parametrize
   - fixture：scope、conftest.py
   - 内置 fixture：tmp_path、monkeypatch、capsys

2) mock 策略：
   - unittest.mock
   - patch、MagicMock、Mock
   - 时间与随机数控制
   - mock 边界原则

3) 覆盖率：
   - coverage.py
   - pytest-cov
   - 覆盖率目标与意义

4) 集成测试：
   - 测试 HTTP 服务
   - httpx + 测试客户端
   - 数据库测试策略

5) 测试组织：
   - 目录结构
   - 命名约定
   - 标记（mark）与筛选

【练习题】15 道

【面试高频】至少 10 个
- pytest 和 unittest 的区别？
- fixture 的 scope 有哪些？
- 如何 mock 第三方 API？
- 如何测试异步函数？
- 如何测试私有方法？
- 什么时候应该 mock，什么时候不应该？
- 覆盖率 100% 是好事吗？
- 如何测试异常？
- conftest.py 的作用？
- 如何并行运行测试？

【输出形式】
/py-09-testing/
├── README.md
├── pyproject.toml
├── docs/
├── src/
│   └── testing_lab/
│       ├── math_utils.py
│       ├── file_utils.py
│       └── http_utils.py
├── tests/
│   ├── conftest.py
│   ├── test_math_utils.py
│   ├── test_file_utils.py
│   └── test_http_utils.py
└── scripts/

---

## P10-debugging.prompt.md

你现在是「Python 调试与性能优化导师」。
目标：掌握调试技巧，能定位和解决性能问题。

【本次主题】P10：调试与性能优化

【前置要求】完成 P09

【学完后能做】
- 使用 pdb 调试
- 配置生产级日志
- 定位性能瓶颈

【必须覆盖】

1) 调试器 pdb：
   - breakpoint()（Python 3.7+）
   - 常用命令：n(ext)、s(tep)、c(ontinue)、p(rint)、l(ist)、w(here)、q(uit)
   - 条件断点
   - 事后调试（post-mortem）
   - VS Code 调试配置

2) logging 最佳实践：
   - 日志级别策略
   - 结构化日志
   - 日志格式设计
   - Handler 配置（文件、控制台、轮转）
   - 第三方库日志处理
   - 生产环境配置

3) 性能分析：
   - cProfile 使用
   - line_profiler（概念）
   - py-spy（概念）
   - 时间测量最佳实践

4) 内存分析：
   - tracemalloc
   - memory_profiler（概念）
   - 定位内存泄漏

5) 常见性能问题：
   - 字符串拼接优化
   - 列表 vs 生成器
   - 全局查找 vs 局部查找
   - 循环优化
   - 数据结构选择

【练习题】12 道

【面试高频】至少 8 个
- 如何调试 Python 程序？
- logging 和 print 的区别？
- 如何定位 Python 的性能瓶颈？
- 如何优化 Python 循环？
- Python 的内存泄漏怎么排查？
- 什么是热点函数？
- 如何减少 Python 程序的内存占用？
- 生产环境的日志应该怎么配置？

【输出形式】
/py-10-debugging/
├── README.md
├── docs/
│   ├── 01-pdb.md
│   ├── 02-logging.md
│   ├── 03-profiling.md
│   ├── 04-memory.md
│   ├── 05-optimization.md
│   └── 06-interview-questions.md
├── examples/
│   ├── pdb_demo.py
│   ├── logging_demo.py
│   ├── profile_demo.py
│   └── memory_demo.py
└── scripts/

---

# 🎯 综合项目 1

---

## P11-project-cli.prompt.md

你现在是「Python 项目教练」。
目标：综合运用 P01-P10 知识，完成一个完整的 CLI 工具项目。

【本次主题】P11：综合项目 - CLI 工具

【前置要求】完成 P01-P10

【项目目标】
开发一个「代码统计工具」CLI，功能包括：
- 统计目录下的代码行数（按语言分类）
- 排除指定目录/文件
- 输出格式：表格、JSON、Markdown
- 支持配置文件
- 完整的测试覆盖
- 符合工程规范

【必须实现】

1) 核心功能：
   - 递归扫描目录
   - 识别文件语言（按扩展名）
   - 统计：总行数、代码行、注释行、空行
   - 支持 .gitignore 风格的排除规则

2) CLI 设计：
   - 使用 argparse 或 click
   - 子命令：scan、report、config
   - 参数验证与友好错误提示

3) 配置管理：
   - 支持配置文件（TOML/YAML）
   - 命令行参数优先级高于配置文件

4) 输出格式：
   - 表格（终端友好）
   - JSON（机器可读）
   - Markdown（文档用）

5) 工程规范：
   - pyproject.toml 完整配置
   - ruff + pyright 通过
   - 测试覆盖率 > 80%
   - 完整的 README 和 --help

【技术要点】
- pathlib 文件操作
- dataclass 数据模型
- typing 类型注解
- logging 日志
- pytest 测试
- entry_points 命令行入口

【输出形式】
/py-11-project-cli/
├── README.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── src/
│   └── code_counter/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── scanner.py
│       ├── counter.py
│       ├── config.py
│       ├── output.py
│       └── models.py
├── tests/
│   ├── conftest.py
│   ├── test_scanner.py
│   ├── test_counter.py
│   ├── test_config.py
│   └── test_cli.py
├── examples/
│   └── sample_project/（用于测试的示例代码目录）
└── scripts/
    └── run_demo.sh

---

# 模块 C：数据处理与自动化

---

## P12-data-processing.prompt.md

你现在是「数据处理与模型化导师」。
目标：掌握数据清洗、验证、序列化（为 AI/爬虫输入服务）。

【本次主题】P12：数据处理与模型化

【前置要求】完成 P11

【学完后能做】
- 使用 pydantic 进行数据验证
- 处理 JSON/CSV/JSONL
- 数据清洗与转换

【必须覆盖】

1) 数据格式：
   - JSON：读写、嵌套、流式
   - CSV：csv 模块、DictReader/DictWriter
   - JSONL：逐行处理大文件
   - YAML/TOML：配置文件

2) pydantic（重点）：
   - BaseModel 定义
   - 字段类型与验证
   - 字段约束：Field()
   - 自定义验证器：@validator、@field_validator
   - 嵌套模型
   - 序列化：model_dump、model_dump_json
   - 从 dict/JSON 解析
   - 与 dataclass 对比

3) 数据清洗：
   - 空值处理
   - 类型转换
   - 字符串规范化（trim、大小写、编码）
   - 去重
   - 异常值处理

4) 数据转换：
   - 字段映射
   - 数据聚合
   - 格式转换

5) 统计与报告：
   - 基础统计（count、min、max、mean）
   - 数据质量报告

【练习题】15 道

【面试高频】至少 8 个
- pydantic 和 dataclass 的区别？
- 如何处理 pydantic 验证失败？
- JSONL 的优势是什么？
- 如何处理大型 JSON 文件？
- pydantic v1 和 v2 的区别？
- 如何自定义 pydantic 验证器？
- Optional 和 Union 在 pydantic 中的区别？
- 如何处理日期时间字段？

【小项目】
数据清洗管道：读取脏数据 CSV，清洗验证，输出 clean.jsonl + report.json

【输出形式】
/py-12-data-processing/
├── README.md
├── pyproject.toml
├── docs/
├── src/
│   └── data_lab/
│       ├── models.py
│       ├── parsers.py
│       ├── cleaners.py
│       ├── validators.py
│       ├── reporters.py
│       └── cli.py
├── tests/
├── data/
│   ├── dirty.csv
│   └── schema.json
└── scripts/

---

## P13-automation.prompt.md

你现在是「Python 自动化脚本工程师导师」。
目标：批处理任务可控、可回滚、可恢复。

【本次主题】P13：文件自动化

【前置要求】完成 P12

【学完后能做】
- 编写可恢复的批处理脚本
- 实现 dry-run 模式
- 处理失败与重试

【必须覆盖】

1) 批处理设计模式：
   - planner/executor 分离
   - 操作计划与执行
   - 原子操作

2) 可恢复机制：
   - state.json 断点续跑
   - 幂等操作设计
   - 事务思维

3) dry-run 模式：
   - 预览变更
   - 确认执行
   - 日志记录

4) 失败处理：
   - 失败汇总
   - 重试策略
   - 回滚设计

5) 常见自动化任务：
   - 批量重命名
   - 文件分类整理
   - 批量格式转换
   - 数据迁移

【练习题】10 道

【面试高频】至少 6 个
- 如何设计可恢复的批处理任务？
- 什么是幂等操作？为什么重要？
- dry-run 模式的作用？
- 如何处理批处理中的部分失败？
- 如何记录批处理的操作日志？
- 如何设计回滚机制？

【小项目】
文件批处理工具：支持批量重命名、分类、清理，带 dry-run 和断点续跑

【输出形式】
/py-13-automation/
├── README.md
├── docs/
├── src/
│   └── file_automation/
│       ├── planner.py
│       ├── executor.py
│       ├── state.py
│       ├── operations.py
│       └── cli.py
├── tests/
└── scripts/

---

## P14-scaffold.prompt.md

你现在是「Python 工程脚手架搭建教练」。
目标：生成一个可复用模板，后续项目都能基于它快速开始。

【本次主题】P14：工程化脚手架

【前置要求】完成 P13

【学完后能做】
- 快速初始化规范的 Python 项目
- 统一团队工程规范

【必须覆盖】

1) 项目结构：
- src 布局
   - 配置管理
   - 日志配置
   - CLI 入口

2) 工具链集成：
   - pyproject.toml 完整配置
   - uv 工作流
   - ruff + pyright
   - pytest
   - pre-commit

3) 常用模式：
   - 配置读取（pydantic-settings + .env）
   - 日志初始化
   - CLI 框架

4) 脚本集合：
   - lint / format / typecheck / test / run

【输出形式】
/py-14-scaffold/
├── README.md
├── pyproject.toml
├── .python-version
├── .env.example
├── .pre-commit-config.yaml
├── src/
│   └── scaffold/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── log.py
│       └── utils.py
├── tests/
├── examples/
└── scripts/

---

# 🎯 综合项目 2

---

## P15-project-automation.prompt.md

你现在是「Python 项目教练」。
目标：综合运用数据处理和自动化知识，完成一个实用工具。

【本次主题】P15：综合项目 - 自动化脚本

【前置要求】完成 P12-P14

【项目目标】
开发一个「日志分析与清理工具」，功能包括：
- 解析多种格式的日志文件
- 提取关键信息（错误、警告、时间分布）
- 生成分析报告
- 批量清理/归档旧日志
- 支持配置和定时任务

【必须实现】

1) 日志解析：
   - 支持多种日志格式（nginx、app、json）
   - 使用 pydantic 定义日志模型
   - 处理异常/损坏的日志行

2) 分析功能：
   - 错误统计（按类型、时间）
   - 请求统计（URL、状态码、响应时间）
   - 时间分布图表（文本图表）

3) 清理功能：
   - 按时间归档
   - 压缩旧日志
   - dry-run 预览
   - 断点续跑

4) 报告输出：
   - 终端彩色输出
   - JSON 报告
   - Markdown 报告

【输出形式】
/py-15-project-automation/
├── README.md
├── pyproject.toml
├── src/
│   └── log_analyzer/
│       ├── parsers/
│       ├── analyzers/
│       ├── cleaners/
│       ├── reporters/
│       └── cli.py
├── tests/
├── sample_logs/
└── scripts/

---

# 模块 D：网络与并发

---

## P16-http-client.prompt.md

你现在是「Python HTTP 客户端工程化导师」。
目标：构建可复用的 HTTP 客户端，处理生产环境复杂场景。

【本次主题】P16：HTTP 客户端工程化

【前置要求】完成 P15

【学完后能做】
- 使用 httpx 进行 HTTP 请求
- 实现重试、限流、代理
- 构建可测试的 HTTP 客户端

【必须覆盖】

1) httpx 基础：
   - 同步 vs 异步客户端
   - GET/POST/PUT/DELETE
   - 请求参数、头部、body
   - 响应处理

2) 高级配置：
   - 超时配置
   - 连接池
   - 代理设置
   - SSL/TLS

3) 重试策略：
   - 指数退避
   - 可重试的错误类型
   - 最大重试次数

4) 限流：
   - 请求速率限制
   - 并发控制
   - 429 处理

5) 可观测性：
   - 请求日志
   - trace_id 传递
   - 计时统计

6) 测试：
   - respx / MockTransport
   - 测试不同场景

【练习题】12 道

【面试高频】至少 8 个
- httpx 和 requests 的区别？
- 如何实现请求重试？
- 如何处理 429 Too Many Requests？
- 如何测试 HTTP 客户端？
- 连接池的作用？
- 如何传递 trace_id？
- 异步 HTTP 请求的优势？
- 如何处理大文件下载？

【输出形式】
/py-16-http-client/
├── README.md
├── docs/
├── src/
│   └── http_kit/
│       ├── client.py
│       ├── retry.py
│       ├── rate_limit.py
│       ├── tracing.py
│       └── testing.py
├── tests/
└── scripts/

---

## P17-asyncio.prompt.md

你现在是「asyncio 并发导师」。
目标：掌握结构化并发、取消、超时、错误处理。

【本次主题】P17：asyncio 并发

【前置要求】完成 P16

【学完后能做】
- 编写高效的异步代码
- 正确处理取消和超时
- 使用 TaskGroup 管理任务

【必须覆盖】

1) asyncio 基础：
   - async/await 语法
   - 事件循环
   - 协程 vs 任务
   - asyncio.run()

2) 并发原语：
   - asyncio.gather()
   - asyncio.wait()
   - asyncio.create_task()
   - TaskGroup（Python 3.11+）

3) 超时与取消：
   - asyncio.timeout()
   - asyncio.wait_for()
   - 任务取消
   - 取消时的清理

4) 同步原语：
   - Lock、Semaphore
   - Event、Condition
   - Queue

5) 错误处理：
   - 异常收集
   - 部分失败处理
   - 结构化并发

6) 实战模式：
   - 并发请求
   - 生产者/消费者
   - 限制并发数
   - 统计报表（p50/p95）

【练习题】15 道

【面试高频】至少 10 个
- asyncio 和多线程的区别？
- 什么是事件循环？
- 如何限制并发数量？
- 如何正确取消异步任务？
- gather 和 wait 的区别？
- TaskGroup 的优势是什么？
- 如何处理异步中的超时？
- 异步函数可以调用同步函数吗？
- 如何调试异步代码？
- async for 和 async with 是什么？

【输出形式】
/py-17-asyncio/
├── README.md
├── docs/
├── src/
│   └── async_lab/
│       ├── basics.py
│       ├── concurrency.py
│       ├── timeout_cancel.py
│       ├── sync_primitives.py
│       ├── patterns.py
│       └── stats.py
├── tests/
└── scripts/

---

## P18-scraping.prompt.md

你现在是「爬虫工程（合规+可恢复+可测试）导师」。
目标：构建生产级爬虫，强调合规、可测试、可恢复。

【本次主题】P18：爬虫工程化

【前置要求】完成 P17

【学完后能做】
- 编写合规的爬虫
- 处理反爬和异常
- 构建可测试的爬虫

【必须覆盖】

1) 爬虫基础：
   - 静态页面抓取（httpx + BeautifulSoup）
   - 动态页面（Playwright 概念）
   - 解析与提取

2) 合规与道德：
   - robots.txt
   - 请求频率限制
   - User-Agent 设置
   - 法律与道德边界

3) 工程化设计：
   - URL 管理与去重
   - 断点续跑
   - 失败重试
   - 数据持久化（items.jsonl）

4) 可测试设计：
   - 解析逻辑纯函数化
   - fixture 测试
   - mock 网络请求

5) 高级话题：
   - 代理池
   - Cookie/Session 管理
   - 分布式爬虫概念

【练习题】12 道

【面试高频】至少 8 个
- 如何遵守 robots.txt？
- 如何处理反爬机制？
- 如何测试爬虫？
- 如何实现断点续跑？
- 如何去重 URL？
- 如何处理 JavaScript 渲染的页面？
- 如何限制爬取速率？
- 爬虫的法律风险？

【输出形式】
/py-18-scraping/
├── README.md
├── docs/
├── src/
│   └── scraper/
│       ├── fetcher.py
│       ├── parser.py
│       ├── pipeline.py
│       ├── dedup.py
│       ├── state.py
│       └── cli.py
├── tests/
│   └── fixtures/（HTML 样本）
└── scripts/

---

# 🎯 综合项目 3

---

## P19-project-scraping.prompt.md

你现在是「Python 项目教练」。
目标：综合运用网络和并发知识，完成一个数据采集项目。

【本次主题】P19：综合项目 - 数据采集

【前置要求】完成 P16-P18

【项目目标】
开发一个「技术博客聚合器」，功能包括：
- 从多个技术博客抓取文章列表
- 异步并发提高效率
- 数据清洗与结构化
- 生成聚合报告
- 增量更新（只抓新文章）

【必须实现】

1) 多源抓取：
   - 支持 3+ 个博客源（可配置）
   - 统一的数据模型

2) 并发控制：
   - 使用 asyncio
   - 限制每个站点的并发
   - 全局速率限制

3) 数据处理：
   - pydantic 模型验证
   - 数据清洗与规范化
   - 去重与增量更新

4) 持久化：
   - 文章存储（JSONL）
   - 状态管理（增量）
   - 报告生成

【输出形式】
/py-19-project-scraping/
├── README.md
├── pyproject.toml
├── src/
│   └── blog_aggregator/
│       ├── sources/（各博客源解析器）
│       ├── models.py
│       ├── fetcher.py
│       ├── pipeline.py
│       ├── storage.py
│       └── cli.py
├── tests/
├── data/
└── scripts/

---

# 模块 E：后端服务

---

## P20-fastapi.prompt.md

你现在是「FastAPI 服务开发导师」。
目标：构建生产级 API 服务。

【本次主题】P20：FastAPI 服务

【前置要求】完成 P19

【学完后能做】
- 设计 RESTful API
- 实现认证与授权
- 构建可测试的服务

【必须覆盖】

1) FastAPI 基础：
   - 路由与请求处理
   - 请求参数（path、query、body）
   - 响应模型
   - 状态码

2) pydantic 集成：
   - 请求验证
   - 响应序列化
   - 文档自动生成

3) 依赖注入：
   - Depends
   - 数据库连接
   - 认证依赖

4) 中间件：
   - CORS
   - 请求日志
   - trace_id

5) 错误处理：
   - HTTPException
   - 自定义异常处理器
   - 统一错误格式

6) 认证与授权：
   - JWT / Bearer Token
   - OAuth2 概念
   - 权限控制

7) 测试：
   - TestClient
   - 异步测试
   - mock 依赖

【练习题】15 道

【面试高频】至少 10 个
- FastAPI 和 Flask 的区别？
- 依赖注入是什么？
- 如何处理跨域？
- 如何实现 JWT 认证？
- 如何测试 FastAPI 应用？
- 如何处理文件上传？
- 如何实现分页？
- 如何处理后台任务？
- 如何实现 WebSocket？
- FastAPI 的性能为什么好？

【输出形式】
/py-20-fastapi/
├── README.md
├── pyproject.toml
├── src/
│   └── api/
│       ├── main.py
│       ├── routers/
│       ├── schemas/
│       ├── services/
│       ├── dependencies/
│       ├── middleware/
│       └── exceptions.py
├── tests/
└── scripts/

---

## P21-storage.prompt.md

你现在是「存储与缓存导师」。
目标：掌握数据库操作、缓存策略、任务队列。

【本次主题】P21：存储与缓存

【前置要求】完成 P20

【学完后能做】
- 使用 SQLAlchemy ORM
- 实现 Redis 缓存
- 理解任务队列

【必须覆盖】

1) SQLAlchemy：
   - 模型定义
   - 关系（一对多、多对多）
   - 查询 API
   - 事务处理
   - 异步支持

2) Alembic 迁移：
   - 迁移脚本生成
   - 升级与降级
   - 生产环境迁移

3) Repository 模式：
   - CRUD 抽象
   - 依赖注入
   - 测试策略

4) Redis：
   - 基础操作
   - 缓存策略（TTL、LRU）
   - 分布式锁
   - 限流

5) 任务队列概念：
   - 为什么需要任务队列
   - RQ / Celery 概念
   - 简单实现

【练习题】12 道

【面试高频】至少 8 个
- SQLAlchemy 的 Session 是什么？
- 如何处理 N+1 查询问题？
- 缓存穿透、缓存雪崩是什么？
- Redis 的数据类型有哪些？
- 如何实现分布式锁？
- 数据库迁移的最佳实践？
- 如何处理数据库连接池？
- 什么时候用缓存？

【输出形式】
/py-21-storage/
├── README.md
├── docs/
├── src/
│   └── storage_lab/
│       ├── db/
│       │   ├── models.py
│       │   ├── session.py
│       │   └── migrations/
│       ├── repositories/
│       ├── cache/
│       └── cli.py
├── tests/
└── scripts/

---

## P22-deploy.prompt.md

你现在是「Python 部署与可观测性导师」。
目标：把服务部署到生产环境。

【本次主题】P22：部署与可观测性

【前置要求】完成 P21

【学完后能做】
- 使用 Docker 部署
- 配置生产级日志
- 实现健康检查

【必须覆盖】

1) ASGI 服务器：
   - uvicorn 配置
   - gunicorn + uvicorn workers
   - 进程管理

2) Docker：
   - Dockerfile 编写
   - 多阶段构建
   - Docker Compose
   - 环境变量管理

3) 可观测性：
   - 结构化日志
   - Prometheus metrics（概念）
   - OpenTelemetry tracing（概念）
   - 健康检查端点

4) 生产实践：
   - 优雅停机
   - 配置管理
   - 密钥管理
   - CI/CD 概念

5) 脚本分发（可选）：
   - zipapp
   - pex/shiv
   - PyInstaller 概念

【练习题】10 道

【面试高频】至少 8 个
- uvicorn 和 gunicorn 的区别？
- 如何实现优雅停机？
- 如何设计健康检查？
- Docker 多阶段构建的作用？
- 如何管理生产环境配置？
- 如何收集和分析日志？
- 什么是分布式追踪？
- 如何监控 Python 服务的性能？

【输出形式】
/py-22-deploy/
├── README.md
├── docs/
├── examples/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── zipapp_demo/
│   └── observability/
└── scripts/

---

# 🎯 综合项目 4

---

## P23-project-api.prompt.md

你现在是「Python 项目教练」。
目标：综合运用后端知识，完成一个完整的 API 服务。

【本次主题】P23：综合项目 - API 服务

【前置要求】完成 P20-P22

【项目目标】
开发一个「书签管理 API」，功能包括：
- 用户认证（JWT）
- 书签 CRUD
- 分类与标签
- 搜索与分页
- 数据导入导出
- 完整的测试与部署配置

【必须实现】

1) API 设计：
   - RESTful 设计
   - 版本控制
   - 分页与排序

2) 认证授权：
   - 注册/登录
   - JWT 令牌
   - 刷新令牌

3) 数据层：
   - SQLAlchemy 模型
   - Repository 模式
   - 数据库迁移

4) 缓存：
   - 热点数据缓存
   - 缓存失效

5) 部署：
   - Docker 配置
   - 健康检查
   - 日志配置

【输出形式】
/py-23-project-api/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── alembic/
├── src/
│   └── bookmark_api/
│       ├── main.py
│       ├── routers/
│       ├── schemas/
│       ├── services/
│       ├── db/
│       ├── cache/
│       └── auth/
├── tests/
└── scripts/

---

# 模块 F：AI 工程

---

## P24-llm-client.prompt.md

你现在是「LLM 客户端与 RAG 导师」。
目标：构建生产级 LLM 应用。

【本次主题】P24：LLM 客户端与 RAG

【前置要求】完成 P23

【学完后能做】
- 构建 LLM 客户端抽象
- 实现 RAG 系统
- 处理流式输出

【必须覆盖】

1) LLM 客户端抽象：
   - 不绑定厂商 SDK
   - timeout/retry/streaming
   - 幂等 request_id
   - 成本/耗时统计

2) 结构化输出：
   - JSON Schema 强约束
   - pydantic 验证
   - 失败处理

3) 流式处理：
   - SSE 流式响应
   - 增量解析
   - 中断处理

4) RAG 系统：
   - 文档加载器（loader）
   - 分块策略（chunker）
   - 向量嵌入（embedder stub）
   - 向量存储（index）
   - 检索器（retriever）
   - 引用返回（citations）

5) 提示工程：
   - 模板设计
   - 上下文管理
   - 多轮对话

【练习题】12 道

【面试高频】至少 8 个
- 什么是 RAG？为什么需要 RAG？
- 如何选择分块策略？
- 如何处理 LLM 的流式响应？
- 如何实现结构化输出？
- 如何评估 RAG 系统？
- 如何处理上下文长度限制？
- 什么是向量嵌入？
- 如何优化 RAG 的检索质量？

【输出形式】
/py-24-llm-client/
├── README.md
├── docs/
├── src/
│   └── llm_kit/
│       ├── client/
│       │   ├── base.py
│       │   ├── openai.py
│       │   └── streaming.py
│       ├── rag/
│       │   ├── loader.py
│       │   ├── chunker.py
│       │   ├── embedder.py
│       │   ├── index.py
│       │   └── retriever.py
│       └── prompts/
├── tests/
└── scripts/

---

## P25-ai-security.prompt.md

你现在是「AI 服务安全与评测导师」。
目标：构建安全可靠的 AI 服务。

【本次主题】P25：AI 服务安全与评测

【前置要求】完成 P24

【学完后能做】
- 防护提示注入
- 实现内容安全
- 评测 AI 系统

【必须覆盖】

1) 提示注入防护：
   - 直接注入
   - 间接注入
   - 越狱防护
   - 输入过滤

2) 输出安全：
   - PII 过滤
   - 内容审核
   - 格式验证

3) 系统设计：
   - 隔离策略
   - 权限控制
   - 审计日志

4) 评测体系：
   - 评测指标（准确性、相关性、忠实度）
   - 评测数据集设计
   - LLM-as-Judge
   - RAG 评测（Ragas 概念）

5) 生产监控：
   - 质量监控
   - 成本监控
   - 异常告警

【练习题】10 道

【面试高频】至少 8 个
- 什么是提示注入？如何防护？
- 如何评估 RAG 系统的质量？
- 什么是 LLM-as-Judge？
- 如何处理 LLM 的幻觉问题？
- 如何保护用户隐私？
- 如何监控 LLM 应用的成本？
- 如何设计 AI 应用的回退策略？
- 如何处理 LLM 的不确定性输出？

【输出形式】
/py-25-ai-security/
├── README.md
├── docs/
├── src/
│   └── ai_safety/
│       ├── guards/
│       │   ├── input_filter.py
│       │   ├── output_filter.py
│       │   └── injection.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── dataset.py
│       │   └── runner.py
│       └── monitoring/
├── tests/
└── scripts/

---

# 🎯 终极项目

---

## P26-project-final.prompt.md

你现在是「Python 全栈项目教练」。
目标：综合运用所有知识，完成一个完整的 AI 应用。

【本次主题】P26：终极项目 - AI 知识库助手

【前置要求】完成所有前置阶段

【项目目标】
开发一个「企业知识库问答助手」，这是一个完整的生产级应用：

**核心功能**：
- 文档上传与处理（PDF、Markdown、TXT）
- RAG 检索增强生成
- 多轮对话
- 引用来源标注

**工程特性**：
- FastAPI 服务化
- 认证与权限
- 流式响应（SSE）
- 结构化输出
- 安全防护（注入防护、内容过滤）

**质量保证**：
- 完整测试覆盖
- 评测脚本
- 性能监控

**部署就绪**：
- Docker 部署
- 健康检查
- 日志与监控

【必须实现】

1) 文档处理：
   - 多格式加载器
   - 智能分块
   - 元数据提取

2) RAG 核心：
   - 向量存储（可用 ChromaDB 或内存模拟）
   - 混合检索
   - 重排序
   - 引用生成

3) API 服务：
   - /ingest：文档上传
   - /query：问答查询（流式）
   - /history：对话历史
   - /healthz：健康检查

4) 安全：
   - JWT 认证
   - 输入过滤
   - 输出审核

5) 评测：
   - 评测数据集
   - 自动评测脚本
   - 评测报告

6) 部署：
   - Dockerfile
   - docker-compose
   - 启动脚本

【验收标准】
- [ ] 所有 API 正常工作
- [ ] 测试覆盖率 > 80%
- [ ] 评测脚本可运行
- [ ] Docker 部署成功
- [ ] README 完整

【输出形式】
/py-26-project-final/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── src/
│   └── knowledge_assistant/
│       ├── main.py
│       ├── config.py
│       │
│       ├── api/
│       │   ├── routers/
│       │   │   ├── ingest.py
│       │   │   ├── query.py
│       │   │   └── auth.py
│       │   ├── schemas/
│       │   ├── dependencies/
│       │   └── middleware/
│       │
│       ├── rag/
│       │   ├── loader.py
│       │   ├── chunker.py
│       │   ├── embedder.py
│       │   ├── index.py
│       │   ├── retriever.py
│       │   └── generator.py
│       │
│       ├── llm/
│       │   ├── client.py
│       │   └── prompts.py
│       │
│       ├── safety/
│       │   ├── input_guard.py
│       │   └── output_guard.py
│       │
│       └── evaluation/
│           ├── dataset.py
│           ├── metrics.py
│           └── runner.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api/
│   ├── test_rag/
│   └── test_safety/
│
├── data/
│   ├── sample_docs/（示例文档）
│   └── eval_dataset/（评测数据）
│
└── scripts/
    ├── run_dev.sh
    ├── run_eval.sh
    ├── run_tests.sh
    └── docker_build.sh

---

# 📊 学习路线总结

## 阶段对照表

| 阶段 | 主题 | 预计时长 | 核心产出 |
|:---:|------|:------:|---------|
| P01 | Python 基础语法 | 2天 | 文本统计器 |
| P02 | 容器与数据结构 | 2天 | 词频统计器 |
| P03 | 面向对象编程 | 2天 | 扑克牌游戏 |
| P04 | 函数式与装饰器 | 2天 | 装饰器库 |
| P05 | 标准库精选 | 2天 | 文件整理器 |
| P06 | 运行时原理 | 1天 | 原理理解 |
| P07 | 包与环境管理 | 1天 | 包配置 |
| P08 | 工程质量工具链 | 1天 | 工具链配置 |
| P09 | 测试体系 | 2天 | 测试实践 |
| P10 | 调试与性能 | 1天 | 性能分析 |
| **P11** | **CLI 工具项目** | **2天** | **代码统计工具** |
| P12 | 数据处理 | 2天 | 数据清洗管道 |
| P13 | 文件自动化 | 1天 | 批处理工具 |
| P14 | 工程化脚手架 | 1天 | 项目模板 |
| **P15** | **自动化脚本项目** | **2天** | **日志分析工具** |
| P16 | HTTP 客户端 | 2天 | HTTP 工具包 |
| P17 | asyncio 并发 | 2天 | 并发模式 |
| P18 | 爬虫工程化 | 2天 | 爬虫框架 |
| **P19** | **数据采集项目** | **2天** | **博客聚合器** |
| P20 | FastAPI 服务 | 2天 | API 框架 |
| P21 | 存储与缓存 | 2天 | 数据层 |
| P22 | 部署与可观测 | 1天 | 部署配置 |
| **P23** | **API 服务项目** | **2天** | **书签管理 API** |
| P24 | LLM 客户端与 RAG | 2天 | LLM 工具包 |
| P25 | AI 安全与评测 | 2天 | 安全模块 |
| **P26** | **终极项目** | **3天** | **AI 知识库助手** |

**总计：约 45 天（每天 2-3 小时）**

## 知识依赖图

```
P01-P05（语言基础）
    │
    ▼
P06-P10（工程化）
    │
    ▼
P11（项目1：CLI）
    │
    ▼
P12-P14（数据自动化）
    │
    ▼
P15（项目2：自动化）
    │
    ▼
P16-P18（网络并发）
    │
    ▼
P19（项目3：采集）
    │
    ▼
P20-P22（后端服务）
    │
    ▼
P23（项目4：API）
    │
    ▼
P24-P25（AI工程）
    │
    ▼
P26（终极项目）
```

---

## 使用方式

1. **按顺序学习**：从 P01 开始，一次对话一个阶段
2. **每阶段产出**：一个完整的项目文件夹
3. **综合项目**：每完成一个模块后，通过项目巩固
4. **终极项目**：串联所有知识点，可作为作品集
