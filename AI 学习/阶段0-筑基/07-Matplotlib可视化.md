# 📊 07 - Matplotlib 可视化

> 用图表讲述数据的故事

---

## 目录

1. [Matplotlib 基础](#1-matplotlib-基础)
2. [常见图表类型](#2-常见图表类型)
3. [图表美化](#3-图表美化)
4. [子图和布局](#4-子图和布局)
5. [Seaborn 简介](#5-seaborn-简介)
6. [练习题](#6-练习题)

---

## 1. Matplotlib 基础

### 1.1 安装和导入

```python
# 安装
# pip install matplotlib

# 导入
import matplotlib.pyplot as plt
import numpy as np

# Jupyter Notebook 中显示图表
# %matplotlib inline
```

### 1.2 基本绘图流程

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建图表
plt.figure(figsize=(10, 6))  # 设置图表大小

# 绑定数据
plt.plot(x, y)

# 添加标签
plt.title('Sine Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()

# 保存图表
# plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')
```

### 1.3 两种绘图风格

```python
# 风格1：pyplot 接口（简单，适合快速绘图）
plt.figure()
plt.plot(x, y)
plt.title('Title')
plt.show()

# 风格2：面向对象接口（灵活，适合复杂图表）
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.set_title('Title')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
```

---

## 2. 常见图表类型

### 2.1 折线图（Line Plot）

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))

# 多条线
plt.plot(x, y1, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linestyle='--', linewidth=2)

# 添加标记
plt.plot(x[::5], y1[::5], 'bo', markersize=8)  # 每隔5个点画一个圆点

plt.title('Trigonometric Functions', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.2 散点图（Scatter Plot）

```python
np.random.seed(42)
n = 100

x = np.random.randn(n)
y = x + np.random.randn(n) * 0.5
colors = np.random.rand(n)
sizes = np.random.rand(n) * 500

plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Color Value')
plt.title('Scatter Plot with Color and Size', fontsize=14)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### 2.3 柱状图（Bar Chart）

```python
categories = ['A', 'B', 'C', 'D', 'E']
values1 = [23, 45, 56, 78, 32]
values2 = [17, 38, 49, 62, 28]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# 并排柱状图
bars1 = ax.bar(x - width/2, values1, width, label='Group 1', color='steelblue')
bars2 = ax.bar(x + width/2, values2, width, label='Group 2', color='coral')

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha='center', va='bottom')

ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Grouped Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
plt.show()
```

### 2.4 直方图（Histogram）

```python
# 生成正态分布数据
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))

# 基本直方图
plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')

# 添加密度曲线
from scipy import stats
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.norm.pdf(x) * len(data) * (8/30), 'r-', linewidth=2, label='Normal PDF')

plt.title('Histogram with Normal Distribution', fontsize=14)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

### 2.5 饼图（Pie Chart）

```python
labels = ['Product A', 'Product B', 'Product C', 'Product D']
sizes = [35, 30, 20, 15]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0, 0)  # 突出第一块

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, explode=explode, labels=labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=90)
ax.set_title('Sales Distribution', fontsize=14)
ax.axis('equal')  # 保持圆形
plt.show()
```

### 2.6 箱线图（Box Plot）

```python
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3', 'Group 4'],
                patch_artist=True)

# 设置颜色
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_title('Box Plot Comparison', fontsize=14)
ax.set_xlabel('Group')
ax.set_ylabel('Value')
ax.grid(True, alpha=0.3)
plt.show()
```

### 2.7 热力图（Heatmap）

```python
# 创建相关性矩阵
data = np.random.rand(10, 10)

plt.figure(figsize=(10, 8))
im = plt.imshow(data, cmap='hot', aspect='auto')
plt.colorbar(im, label='Value')
plt.title('Heatmap', fontsize=14)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

---

## 3. 图表美化

### 3.1 颜色和样式

```python
# 颜色指定方式
plt.plot(x, y, color='red')           # 颜色名
plt.plot(x, y, color='#FF5733')       # 十六进制
plt.plot(x, y, color=(0.1, 0.2, 0.5)) # RGB 元组
plt.plot(x, y, color='C0')            # 默认颜色循环

# 线型
plt.plot(x, y, linestyle='-')   # 实线
plt.plot(x, y, linestyle='--')  # 虚线
plt.plot(x, y, linestyle='-.')  # 点划线
plt.plot(x, y, linestyle=':')   # 点线

# 标记
plt.plot(x, y, marker='o')  # 圆点
plt.plot(x, y, marker='s')  # 方形
plt.plot(x, y, marker='^')  # 三角形
plt.plot(x, y, marker='*')  # 星形

# 组合写法
plt.plot(x, y, 'ro--')  # 红色圆点虚线
plt.plot(x, y, 'b^-')   # 蓝色三角实线
```

### 3.2 图例和标签

```python
x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')

# 设置标题和标签
ax.set_title('Trigonometric Functions', fontsize=16, fontweight='bold')
ax.set_xlabel('X axis', fontsize=12)
ax.set_ylabel('Y axis', fontsize=12)

# 设置图例
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# 设置刻度
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

plt.show()
```

### 3.3 使用样式

```python
# 查看可用样式
print(plt.style.available)

# 使用预设样式
plt.style.use('seaborn')
# plt.style.use('ggplot')
# plt.style.use('dark_background')

# 临时使用样式
with plt.style.context('seaborn'):
    plt.plot(x, np.sin(x))
    plt.show()
```

---

## 4. 子图和布局

### 4.1 基本子图

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(0, 10, 100)

# 左上
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin(x)')

# 右上
axes[0, 1].plot(x, np.cos(x), 'r')
axes[0, 1].set_title('Cos(x)')

# 左下
axes[1, 0].plot(x, np.tan(x), 'g')
axes[1, 0].set_title('Tan(x)')
axes[1, 0].set_ylim(-5, 5)

# 右下
axes[1, 1].plot(x, np.exp(-x/5), 'm')
axes[1, 1].set_title('Exp(-x/5)')

plt.tight_layout()  # 自动调整间距
plt.show()
```

### 4.2 不同大小的子图

```python
fig = plt.figure(figsize=(12, 8))

# 使用 GridSpec
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :])  # 第一行，跨所有列
ax2 = fig.add_subplot(gs[1, 0])  # 第二行第一列
ax3 = fig.add_subplot(gs[1, 1])  # 第二行第二列
ax4 = fig.add_subplot(gs[1, 2])  # 第二行第三列

ax1.plot(np.random.randn(100))
ax1.set_title('Full Width')

ax2.bar([1, 2, 3], [4, 5, 6])
ax2.set_title('Bar')

ax3.scatter(np.random.rand(20), np.random.rand(20))
ax3.set_title('Scatter')

ax4.hist(np.random.randn(100), bins=20)
ax4.set_title('Histogram')

plt.tight_layout()
plt.show()
```

### 4.3 双 Y 轴

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x / 3)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 左侧 Y 轴
color1 = 'tab:blue'
ax1.set_xlabel('X')
ax1.set_ylabel('sin(x)', color=color1)
ax1.plot(x, y1, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# 右侧 Y 轴
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('exp(x/3)', color=color2)
ax2.plot(x, y2, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Dual Y-axis Plot')
plt.tight_layout()
plt.show()
```

---

## 5. Seaborn 简介

### 5.1 Seaborn 基础

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 设置风格
sns.set_theme(style="whitegrid")

# 加载示例数据
tips = sns.load_dataset("tips")
print(tips.head())
```

### 5.2 常用图表

```python
# 分布图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 直方图 + KDE
sns.histplot(tips['total_bill'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Total Bill')

# 箱线图
sns.boxplot(x='day', y='total_bill', data=tips, ax=axes[0, 1])
axes[0, 1].set_title('Total Bill by Day')

# 小提琴图
sns.violinplot(x='day', y='total_bill', data=tips, ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot')

# 散点图 + 回归线
sns.regplot(x='total_bill', y='tip', data=tips, ax=axes[1, 1])
axes[1, 1].set_title('Bill vs Tip')

plt.tight_layout()
plt.show()
```

### 5.3 热力图（相关性矩阵）

```python
# 计算相关性
corr = tips[['total_bill', 'tip', 'size']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```

### 5.4 分类图

```python
# 按类别分组的图
g = sns.catplot(x='day', y='total_bill', hue='sex', col='time',
                data=tips, kind='bar', height=5, aspect=0.7)
g.fig.suptitle('Tips by Day, Time, and Gender', y=1.02)
plt.show()
```

---

## 6. 练习题

### 基础练习

1. 绑制一个折线图，显示 y = x² 在 [-10, 10] 区间的曲线
2. 创建一个包含 4 个子图的图表，分别展示正弦、余弦、正切、指数函数
3. 绑制一个柱状图，比较不同产品的销售额

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 折线图
x = np.linspace(-10, 10, 100)
y = x ** 2

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('y = x²')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.show()

# 2. 四个子图
x = np.linspace(0, 2*np.pi, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, np.sin(x), 'b')
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, np.cos(x), 'r')
axes[0, 1].set_title('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x, np.tan(x), 'g')
axes[1, 0].set_title('tan(x)')
axes[1, 0].set_ylim(-5, 5)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, np.exp(x), 'm')
axes[1, 1].set_title('exp(x)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. 柱状图
products = ['Product A', 'Product B', 'Product C', 'Product D']
sales = [120, 85, 150, 95]

plt.figure(figsize=(10, 6))
bars = plt.bar(products, sales, color=['steelblue', 'coral', 'green', 'purple'])

# 添加数值标签
for bar, sale in zip(bars, sales):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{sale}', ha='center', fontsize=12)

plt.title('Sales by Product', fontsize=14)
plt.xlabel('Product')
plt.ylabel('Sales')
plt.ylim(0, 180)
plt.show()
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [08-数学直觉.md](./08-数学直觉.md)

