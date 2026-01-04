# 快速排序 (Quick Sort)

> 分区征服：平均性能最优的通用排序算法

## 📦 快速使用

```typescript
import { sort, sortInPlace, sortRobust, meta } from './src/index';
import { numberAsc, byField, compose } from '../公共库/src/比较器';

// 基础排序
const numbers = [38, 27, 43, 3, 9, 82, 10];
const sorted = sort(numbers, numberAsc);
// [3, 9, 10, 27, 38, 43, 82]

// 原地排序（性能最优）
const arr = [5, 3, 8, 4, 2];
sortInPlace(arr, numberAsc);
// arr 现在是 [2, 3, 4, 5, 8]

// 健壮版本（带优化，适合生产环境）
const largeArray = Array.from({ length: 100000 }, () => Math.random());
const result = sortRobust(largeArray, numberAsc);
```

## 🔧 API

### `sort<T>(arr, cmp): T[]`
返回排序后的新数组，不修改原数组。

### `sortInPlace<T>(arr, cmp): T[]`
原地排序，返回同一引用。

### `sortRobust<T>(arr, cmp): T[]`
健壮版本：三数取中 + 尾递归优化 + 小数组插入排序。

### `partition<T>(arr, left, right, cmp): number`
Lomuto 分区，返回 pivot 索引（可单独使用）。

### `meta`
算法元信息（复杂度、稳定性等）。

## 📊 复杂度

| 指标 | 值 |
|------|-----|
| 时间（最好/平均） | O(n log n) |
| 时间（最坏） | O(n²) ⚠️ |
| 空间 | O(log n) 栈空间 |
| 稳定性 | ❌ 不稳定 |
| 原地 | ✅ 原地 |

## 📁 文件结构

```
快速排序/
├── README.md
├── src/
│   ├── index.ts    # 核心实现
│   └── demo.ts     # 使用示例
└── test/
    └── index.test.ts
```

