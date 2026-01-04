# 03. datetime - 日期时间

## 本节目标

- 掌握 date、time、datetime、timedelta
- 熟练格式化和解析日期
- 理解时区处理

---

## 核心类型

```python
from datetime import date, time, datetime, timedelta

# date - 日期
d = date(2024, 1, 15)
d = date.today()

# time - 时间
t = time(14, 30, 45)
t = time(14, 30, 45, 123456)  # 微秒

# datetime - 日期时间
dt = datetime(2024, 1, 15, 14, 30, 45)
dt = datetime.now()
dt = datetime.today()

# timedelta - 时间差
delta = timedelta(days=7)
delta = timedelta(hours=2, minutes=30)
```

---

## 创建日期时间

```python
from datetime import datetime, date

# 当前时间
now = datetime.now()
today = date.today()

# 从时间戳
dt = datetime.fromtimestamp(1705312245)

# 从字符串（ISO 格式）
dt = datetime.fromisoformat("2024-01-15T14:30:45")
d = date.fromisoformat("2024-01-15")

# 从字符串（自定义格式）
dt = datetime.strptime("15/01/2024", "%d/%m/%Y")
```

### JS 对照

```javascript
// JavaScript
const now = new Date();
const dt = new Date('2024-01-15T14:30:45');
const ts = new Date(1705312245 * 1000);
```

---

## 日期时间属性

```python
from datetime import datetime

dt = datetime(2024, 1, 15, 14, 30, 45)

print(dt.year)        # 2024
print(dt.month)       # 1
print(dt.day)         # 15
print(dt.hour)        # 14
print(dt.minute)      # 30
print(dt.second)      # 45
print(dt.microsecond) # 0

print(dt.weekday())   # 0（周一）
print(dt.isoweekday()) # 1（ISO 周一=1）

# 获取 date 和 time 部分
print(dt.date())      # 2024-01-15
print(dt.time())      # 14:30:45
```

---

## 格式化输出

### strftime - 格式化为字符串

```python
from datetime import datetime

dt = datetime(2024, 1, 15, 14, 30, 45)

# 常用格式
print(dt.strftime("%Y-%m-%d"))          # 2024-01-15
print(dt.strftime("%Y/%m/%d %H:%M:%S")) # 2024/01/15 14:30:45
print(dt.strftime("%B %d, %Y"))         # January 15, 2024
print(dt.strftime("%A"))                # Monday

# ISO 格式
print(dt.isoformat())  # 2024-01-15T14:30:45
```

### 常用格式符

| 格式符 | 含义 | 示例 |
|--------|------|------|
| `%Y` | 四位年份 | 2024 |
| `%m` | 月份（补零） | 01 |
| `%d` | 日（补零） | 15 |
| `%H` | 24小时制 | 14 |
| `%I` | 12小时制 | 02 |
| `%M` | 分钟 | 30 |
| `%S` | 秒 | 45 |
| `%p` | AM/PM | PM |
| `%A` | 星期全名 | Monday |
| `%a` | 星期缩写 | Mon |
| `%B` | 月份全名 | January |
| `%b` | 月份缩写 | Jan |

---

## 解析字符串

### strptime - 字符串转日期

```python
from datetime import datetime

# 解析
dt = datetime.strptime("2024-01-15", "%Y-%m-%d")
dt = datetime.strptime("15/01/2024 14:30", "%d/%m/%Y %H:%M")
dt = datetime.strptime("Jan 15, 2024", "%b %d, %Y")
```

---

## 日期计算

### timedelta

```python
from datetime import datetime, timedelta

now = datetime.now()

# 加减
tomorrow = now + timedelta(days=1)
last_week = now - timedelta(weeks=1)
later = now + timedelta(hours=2, minutes=30)

# 日期差
d1 = datetime(2024, 1, 15)
d2 = datetime(2024, 1, 10)
diff = d1 - d2
print(diff.days)          # 5
print(diff.total_seconds()) # 432000.0
```

### 月份计算

timedelta 不支持月份，需要手动处理：

```python
from datetime import datetime

def add_months(dt, months):
    month = dt.month + months
    year = dt.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(dt.day, [31,28,31,30,31,30,31,31,30,31,30,31][month-1])
    return dt.replace(year=year, month=month, day=day)

# 或使用 dateutil 库（第三方）
# from dateutil.relativedelta import relativedelta
# dt + relativedelta(months=1)
```

---

## 日期比较

```python
from datetime import datetime

d1 = datetime(2024, 1, 15)
d2 = datetime(2024, 1, 10)

print(d1 > d2)   # True
print(d1 == d2)  # False
print(d1 - d2)   # 5 days, 0:00:00

# 排序
dates = [
    datetime(2024, 1, 15),
    datetime(2024, 1, 10),
    datetime(2024, 1, 20),
]
sorted_dates = sorted(dates)
```

---

## 时区处理

### timezone（标准库）

```python
from datetime import datetime, timezone, timedelta

# UTC
utc_now = datetime.now(timezone.utc)
print(utc_now)

# 自定义时区
cst = timezone(timedelta(hours=8))  # UTC+8
local_time = datetime.now(cst)

# 转换时区
utc_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
local_time = utc_time.astimezone(cst)
```

### zoneinfo（Python 3.9+）

```python
from datetime import datetime
from zoneinfo import ZoneInfo

# 使用时区名称
utc = ZoneInfo("UTC")
shanghai = ZoneInfo("Asia/Shanghai")
new_york = ZoneInfo("America/New_York")

# 创建带时区的时间
dt = datetime(2024, 1, 15, 12, 0, tzinfo=shanghai)
print(dt)  # 2024-01-15 12:00:00+08:00

# 转换时区
dt_ny = dt.astimezone(new_york)
print(dt_ny)  # 2024-01-14 23:00:00-05:00
```

### naive vs aware

```python
from datetime import datetime, timezone

# naive（无时区信息）
naive = datetime.now()
print(naive.tzinfo)  # None

# aware（有时区信息）
aware = datetime.now(timezone.utc)
print(aware.tzinfo)  # UTC

# 不能混合比较
# naive > aware  # TypeError!
```

---

## 时间戳转换

```python
from datetime import datetime, timezone

# datetime 转时间戳
dt = datetime.now()
timestamp = dt.timestamp()
print(timestamp)  # 1705312245.123456

# 时间戳转 datetime
dt = datetime.fromtimestamp(timestamp)
dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
```

---

## 实际应用

### 计算年龄

```python
from datetime import date

def calculate_age(birthday):
    today = date.today()
    age = today.year - birthday.year
    if (today.month, today.day) < (birthday.month, birthday.day):
        age -= 1
    return age

birthday = date(1990, 5, 15)
print(calculate_age(birthday))
```

### 日期范围

```python
from datetime import date, timedelta

def date_range(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

for d in date_range(date(2024, 1, 1), date(2024, 1, 5)):
    print(d)
```

---

## 本节要点

1. `datetime.now()` 当前时间，`date.today()` 当前日期
2. `strftime()` 格式化，`strptime()` 解析
3. `timedelta` 进行日期计算
4. `timezone` 或 `zoneinfo` 处理时区
5. aware datetime 有时区，naive 没有


