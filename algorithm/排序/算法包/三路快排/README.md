# 三路快速排序 (3-Way Quick Sort)

> 大量重复元素的克星：Dijkstra 三路分区

## 📦 快速使用

```typescript
import { sort, sortInPlace, meta } from './src/index';
import { numberAsc } from '../公共库/src/比较器';

// 大量重复元素
const numbers = [5, 3, 5, 5, 2, 5, 1, 5, 4, 5];
const sorted = sort(numbers, numberAsc);
// [1, 2, 3, 4, 5, 5, 5, 5, 5, 5]

// 状态码排序（重复率高）
const statusOrder = { pending: 0, completed: 1, failed: 2 };
const tasks = [
  { id: 1, status: 'pending' },
  { id: 2, status: 'completed' },
  { id: 3, status: 'pending' },
];
const sortedTasks = sort(tasks, (a, b) =>
  statusOrder[a.status] - statusOrder[b.status]
);
```

## 🔧 API

### `sort<T>(arr, cmp): T[]`
返回排序后的新数组，不修改原数组。

### `sortInPlace<T>(arr, cmp): T[]`
原地排序，返回同一引用。

### `meta`
算法元信息。

## 📊 复杂度

| 指标 | 标准快排 | 三路快排 |
|------|---------|---------|
| 最好/平均 | O(n log n) | O(n log n) |
| 最坏（全相同） | O(n²) ⚠️ | **O(n)** ⭐ |
| 空间 | O(log n) | O(log n) |
| 稳定性 | ❌ | ❌ |

## 📁 文件结构

```
三路快排/
├── README.md
├── src/
│   ├── index.ts    # 核心实现
│   └── demo.ts     # 使用示例
└── test/
    └── index.test.ts
```

