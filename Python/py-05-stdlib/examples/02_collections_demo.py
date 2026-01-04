#!/usr/bin/env python3
"""collections 模块演示"""

from collections import Counter, defaultdict, deque, namedtuple, ChainMap, OrderedDict


def demo_counter():
    """Counter 计数器"""
    print("=" * 50)
    print("1. Counter 计数器")
    print("=" * 50)

    # 创建
    words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    counter = Counter(words)
    print(f"Counter: {counter}")

    # 字符串计数
    char_counter = Counter("hello world")
    print(f"字符计数: {char_counter}")

    # 最常见元素
    print(f"最常见的 2 个: {counter.most_common(2)}")

    # 访问计数
    print(f"apple 出现次数: {counter['apple']}")
    print(f"不存在的元素: {counter['orange']}")  # 返回 0

    # 更新
    counter.update(["banana", "orange"])
    print(f"更新后: {counter}")

    # 运算
    c1 = Counter(a=3, b=1)
    c2 = Counter(a=1, b=2)
    print(f"c1 + c2: {c1 + c2}")
    print(f"c1 - c2: {c1 - c2}")


def demo_defaultdict():
    """defaultdict 默认值字典"""
    print("\n" + "=" * 50)
    print("2. defaultdict 默认值字典")
    print("=" * 50)

    # int 作为默认值（计数）
    counter = defaultdict(int)
    for word in ["apple", "banana", "apple"]:
        counter[word] += 1
    print(f"计数器: {dict(counter)}")

    # list 作为默认值（分组）
    groups = defaultdict(list)
    students = [("A班", "张三"), ("B班", "李四"), ("A班", "王五")]
    for cls, name in students:
        groups[cls].append(name)
    print(f"分组: {dict(groups)}")

    # set 作为默认值（去重分组）
    unique_groups = defaultdict(set)
    tags = [("python", "web"), ("python", "ai"), ("js", "web")]
    for lang, tag in tags:
        unique_groups[lang].add(tag)
    print(f"去重分组: {dict(unique_groups)}")

    # 自定义默认值
    custom = defaultdict(lambda: "N/A")
    custom["name"] = "Alice"
    print(f"自定义默认值: name={custom['name']}, age={custom['age']}")


def demo_deque():
    """deque 双端队列"""
    print("\n" + "=" * 50)
    print("3. deque 双端队列")
    print("=" * 50)

    # 创建
    d = deque([1, 2, 3])
    print(f"创建: {d}")

    # 右端操作
    d.append(4)
    print(f"append(4): {d}")
    d.pop()
    print(f"pop(): {d}")

    # 左端操作
    d.appendleft(0)
    print(f"appendleft(0): {d}")
    d.popleft()
    print(f"popleft(): {d}")

    # 旋转
    d = deque([1, 2, 3, 4, 5])
    d.rotate(2)
    print(f"rotate(2): {d}")
    d.rotate(-2)
    print(f"rotate(-2): {d}")

    # 固定长度
    recent = deque(maxlen=3)
    for i in range(5):
        recent.append(i)
        print(f"  append({i}): {list(recent)}")


def demo_namedtuple():
    """namedtuple 命名元组"""
    print("\n" + "=" * 50)
    print("4. namedtuple 命名元组")
    print("=" * 50)

    # 定义
    Point = namedtuple("Point", ["x", "y"])

    # 创建
    p = Point(10, 20)
    print(f"Point: {p}")

    # 访问
    print(f"p.x: {p.x}, p.y: {p.y}")
    print(f"p[0]: {p[0]}, p[1]: {p[1]}")

    # 解包
    x, y = p
    print(f"解包: x={x}, y={y}")

    # 转为字典
    print(f"_asdict(): {p._asdict()}")

    # 替换
    p2 = p._replace(x=100)
    print(f"_replace(x=100): {p2}")

    # 带默认值
    Person = namedtuple("Person", ["name", "age", "city"], defaults=["Unknown"])
    person = Person("Alice", 25)
    print(f"带默认值: {person}")


def demo_chainmap():
    """ChainMap 字典链"""
    print("\n" + "=" * 50)
    print("5. ChainMap 字典链")
    print("=" * 50)

    # 配置优先级
    defaults = {"debug": False, "log_level": "INFO", "timeout": 30}
    user_config = {"debug": True}
    cli_args = {"log_level": "DEBUG"}

    config = ChainMap(cli_args, user_config, defaults)

    print(f"debug: {config['debug']}")      # 来自 user_config
    print(f"log_level: {config['log_level']}")  # 来自 cli_args
    print(f"timeout: {config['timeout']}")  # 来自 defaults

    # 查看所有 key
    print(f"所有配置项: {list(config.keys())}")


def demo_ordereddict():
    """OrderedDict 有序字典"""
    print("\n" + "=" * 50)
    print("6. OrderedDict 有序字典")
    print("=" * 50)

    # Python 3.7+ dict 已有序，OrderedDict 提供额外功能
    od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
    print(f"OrderedDict: {od}")

    # 移动到末尾
    od.move_to_end("a")
    print(f"move_to_end('a'): {od}")

    # 移动到开头
    od.move_to_end("c", last=False)
    print(f"move_to_end('c', last=False): {od}")

    # 弹出最后/最前
    od.popitem(last=True)
    print(f"popitem(last=True): {od}")


if __name__ == "__main__":
    demo_counter()
    demo_defaultdict()
    demo_deque()
    demo_namedtuple()
    demo_chainmap()
    demo_ordereddict()

    print("\n✅ collections 演示完成!")


