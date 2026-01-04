#!/usr/bin/env python3
"""datetime 模块演示"""

from datetime import date, time, datetime, timedelta, timezone
import sys

# Python 3.9+ 才有 zoneinfo
if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
    HAS_ZONEINFO = True
else:
    HAS_ZONEINFO = False


def demo_basic_types():
    """基本类型"""
    print("=" * 50)
    print("1. 基本类型")
    print("=" * 50)

    # date
    d = date(2024, 1, 15)
    print(f"date: {d}")
    print(f"date.today(): {date.today()}")

    # time
    t = time(14, 30, 45)
    print(f"time: {t}")

    # datetime
    dt = datetime(2024, 1, 15, 14, 30, 45)
    print(f"datetime: {dt}")
    print(f"datetime.now(): {datetime.now()}")

    # timedelta
    delta = timedelta(days=7, hours=2)
    print(f"timedelta: {delta}")


def demo_datetime_creation():
    """创建日期时间"""
    print("\n" + "=" * 50)
    print("2. 创建日期时间")
    print("=" * 50)

    # 当前时间
    now = datetime.now()
    print(f"now(): {now}")

    # 从时间戳
    ts = 1705312245
    dt = datetime.fromtimestamp(ts)
    print(f"fromtimestamp({ts}): {dt}")

    # 从 ISO 格式字符串
    dt = datetime.fromisoformat("2024-01-15T14:30:45")
    print(f"fromisoformat: {dt}")

    # 从自定义格式
    dt = datetime.strptime("15/01/2024 14:30", "%d/%m/%Y %H:%M")
    print(f"strptime: {dt}")


def demo_attributes():
    """日期时间属性"""
    print("\n" + "=" * 50)
    print("3. 日期时间属性")
    print("=" * 50)

    dt = datetime(2024, 1, 15, 14, 30, 45)

    print(f"datetime: {dt}")
    print(f"year: {dt.year}")
    print(f"month: {dt.month}")
    print(f"day: {dt.day}")
    print(f"hour: {dt.hour}")
    print(f"minute: {dt.minute}")
    print(f"second: {dt.second}")
    print(f"weekday(): {dt.weekday()} (0=周一)")
    print(f"isoweekday(): {dt.isoweekday()} (1=周一)")
    print(f"date(): {dt.date()}")
    print(f"time(): {dt.time()}")


def demo_formatting():
    """格式化"""
    print("\n" + "=" * 50)
    print("4. 格式化")
    print("=" * 50)

    dt = datetime(2024, 1, 15, 14, 30, 45)

    print(f"原始: {dt}")
    print(f"isoformat(): {dt.isoformat()}")
    print(f"%Y-%m-%d: {dt.strftime('%Y-%m-%d')}")
    print(f"%Y/%m/%d %H:%M:%S: {dt.strftime('%Y/%m/%d %H:%M:%S')}")
    print(f"%B %d, %Y: {dt.strftime('%B %d, %Y')}")
    print(f"%A: {dt.strftime('%A')}")


def demo_calculations():
    """日期计算"""
    print("\n" + "=" * 50)
    print("5. 日期计算")
    print("=" * 50)

    now = datetime.now()

    # 加减
    tomorrow = now + timedelta(days=1)
    print(f"明天: {tomorrow.date()}")

    last_week = now - timedelta(weeks=1)
    print(f"上周: {last_week.date()}")

    later = now + timedelta(hours=2, minutes=30)
    print(f"2.5 小时后: {later.time()}")

    # 日期差
    d1 = datetime(2024, 1, 15)
    d2 = datetime(2024, 1, 10)
    diff = d1 - d2
    print(f"日期差: {diff}")
    print(f"相差天数: {diff.days}")
    print(f"总秒数: {diff.total_seconds()}")


def demo_comparison():
    """日期比较"""
    print("\n" + "=" * 50)
    print("6. 日期比较")
    print("=" * 50)

    d1 = datetime(2024, 1, 15)
    d2 = datetime(2024, 1, 10)
    d3 = datetime(2024, 1, 20)

    print(f"d1 = {d1.date()}")
    print(f"d2 = {d2.date()}")
    print(f"d3 = {d3.date()}")
    print(f"d1 > d2: {d1 > d2}")
    print(f"d1 < d3: {d1 < d3}")

    # 排序
    dates = [d1, d2, d3]
    sorted_dates = sorted(dates)
    print(f"排序后: {[d.date() for d in sorted_dates]}")


def demo_timezone():
    """时区处理"""
    print("\n" + "=" * 50)
    print("7. 时区处理")
    print("=" * 50)

    # UTC
    utc_now = datetime.now(timezone.utc)
    print(f"UTC 时间: {utc_now}")

    # 自定义时区
    cst = timezone(timedelta(hours=8))
    local_time = datetime.now(cst)
    print(f"UTC+8 时间: {local_time}")

    # naive vs aware
    naive = datetime.now()
    print(f"naive datetime (无时区): {naive}, tzinfo={naive.tzinfo}")
    print(f"aware datetime (有时区): {utc_now}, tzinfo={utc_now.tzinfo}")

    # zoneinfo (Python 3.9+)
    if HAS_ZONEINFO:
        shanghai = ZoneInfo("Asia/Shanghai")
        shanghai_time = datetime.now(shanghai)
        print(f"上海时间: {shanghai_time}")

        # 时区转换
        ny = ZoneInfo("America/New_York")
        ny_time = shanghai_time.astimezone(ny)
        print(f"纽约时间: {ny_time}")
    else:
        print("(zoneinfo 需要 Python 3.9+)")


def demo_timestamp():
    """时间戳转换"""
    print("\n" + "=" * 50)
    print("8. 时间戳转换")
    print("=" * 50)

    # datetime 转时间戳
    dt = datetime.now()
    ts = dt.timestamp()
    print(f"datetime: {dt}")
    print(f"时间戳: {ts}")

    # 时间戳转 datetime
    dt2 = datetime.fromtimestamp(ts)
    print(f"转回 datetime: {dt2}")


if __name__ == "__main__":
    demo_basic_types()
    demo_datetime_creation()
    demo_attributes()
    demo_formatting()
    demo_calculations()
    demo_comparison()
    demo_timezone()
    demo_timestamp()

    print("\n✅ datetime 演示完成!")


