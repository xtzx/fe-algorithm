# 归并排序 (Merge Sort)

> 分治典范：稳定的 O(n log n) 排序算法

## 📦 快速使用

```typescript
import { sort, sortInPlace, sortIterative, meta } from './src/index';
import { numberAsc, byField, compose } from '../公共库/src/比较器';

// 基础排序
const numbers = [38, 27, 43, 3, 9, 82, 10];
const sorted = sort(numbers, numberAsc);
// [3, 9, 10, 27, 38, 43, 82]

// 对象排序（稳定性保证）
const users = [
  { name: 'Bob', age: 25 },
  { name: 'Alice', age: 25 },
  { name: 'Charlie', age: 30 },
];
const byAge = byField('age', numberAsc);
const sortedUsers = sort(users, byAge);
// Alice 和 Bob 保持原顺序（都是 25 岁）

// 迭代版（避免栈溢出）
const largeArray = Array.from({ length: 100000 }, () => Math.random());
const result = sortIterative(largeArray, numberAsc);
```

## 🔧 API

### `sort<T>(arr, cmp): T[]`
返回排序后的新数组，不修改原数组。

### `sortInPlace<T>(arr, cmp): T[]`
原地排序（使用辅助数组），返回同一引用。

### `sortIterative<T>(arr, cmp): T[]`
迭代版（自底向上），无递归栈风险。

### `merge<T>(left, right, cmp): T[]`
合并两个已排序数组（可单独使用）。

### `meta`
算法元信息（复杂度、稳定性等）。

## 📊 复杂度

| 指标 | 值 |
|------|-----|
| 时间（最好/平均/最坏） | O(n log n) |
| 空间 | O(n) |
| 稳定性 | ✅ 稳定 |
| 原地 | ❌ 需要辅助空间 |

## 📁 文件结构

```
归并排序/
├── README.md
├── src/
│   ├── index.ts    # 核心实现
│   └── demo.ts     # 使用示例
└── test/
    └── index.test.ts
```

