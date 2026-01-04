# 堆排序 (Heap Sort)

> 基于堆数据结构：原地、稳定的 O(n log n) 保证

## 📦 快速使用

```typescript
import { sort, sortInPlace, heapifyDown, meta } from './src/index';
import { numberAsc, byField, reverse } from '../公共库/src/比较器';

// 基础排序
const numbers = [38, 27, 43, 3, 9, 82, 10];
const sorted = sort(numbers, numberAsc);
// [3, 9, 10, 27, 38, 43, 82]

// 原地排序
const arr = [5, 3, 8, 4, 2];
sortInPlace(arr, numberAsc);
// arr 现在是 [2, 3, 4, 5, 8]

// TopK（维护小堆找最大的 K 个）
const topK = findTopK(numbers, 3, numberAsc);
// 最大的 3 个元素
```

## 🔧 API

### `sort<T>(arr, cmp): T[]`
返回排序后的新数组，不修改原数组。

### `sortInPlace<T>(arr, cmp): T[]`
原地排序，返回同一引用。

### `heapifyDown<T>(arr, size, i, cmp): void`
向下堆化（单独使用，用于优先队列实现）。

### `buildHeap<T>(arr, cmp): void`
建堆（自底向上，O(n)）。

### `meta`
算法元信息。

## 📊 复杂度

| 指标 | 值 |
|------|-----|
| 时间（所有情况） | O(n log n) ⭐ |
| 空间 | O(1) 原地 |
| 稳定性 | ❌ 不稳定 |
| 原地 | ✅ 原地 |

## 📁 文件结构

```
堆排序/
├── README.md
├── src/
│   ├── index.ts    # 核心实现
│   └── demo.ts     # 使用示例
└── test/
    └── index.test.ts
```

