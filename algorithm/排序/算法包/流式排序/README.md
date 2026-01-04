# 流式排序

> 处理持续到来的数据流，维护实时有序状态

## 📖 功能概述

- **SortedWindow** - 维护固定容量的有序窗口（Top K）
- **OnlineMedian** - 实时计算数据流中位数（双堆方案）

## 🚀 使用方法

### 有序窗口（Top K）

```typescript
import { SortedWindow } from './src';

// 维护 Top 5 最大值
const window = new SortedWindow<number>(5, (a, b) => a - b);

window.add(10);
window.add(3);
window.add(7);
window.add(15);
window.add(8);
window.add(20); // 3 被淘汰

console.log(window.toArray()); // [7, 8, 10, 15, 20]
console.log(window.get(0));    // 7 (最小)
console.log(window.get(4));    // 20 (最大)
```

### 在线中位数

```typescript
import { OnlineMedian } from './src';

const median = new OnlineMedian();

median.add(1);
console.log(median.getMedian()); // 1

median.add(2);
console.log(median.getMedian()); // 1.5

median.add(3);
console.log(median.getMedian()); // 2
```

## 📊 复杂度

| 数据结构 | 插入 | 查询 | 空间 |
|---------|------|------|------|
| SortedWindow | O(k) | O(1) | O(k) |
| OnlineMedian | O(log n) | O(1) | O(n) |

## 📁 目录结构

```
流式排序/
├── README.md
├── src/
│   ├── index.ts          # 导出入口
│   ├── sortedWindow.ts   # 有序窗口实现
│   └── onlineMedian.ts   # 在线中位数实现
└── test/
    └── index.test.ts     # 测试文件
```

## 🔗 LeetCode 相关

| 题号 | 题目 | 难度 |
|:----:|------|:----:|
| 295 | 数据流的中位数 | H |
| 703 | 数据流中的第 K 大元素 | E |
| 480 | 滑动窗口中位数 | H |

