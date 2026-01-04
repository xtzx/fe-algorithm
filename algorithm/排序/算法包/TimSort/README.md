# TimSort

> Python/Java 默认排序：近乎有序数据的王者，稳定排序的首选

## 📦 快速使用

```typescript
import { sort, meta } from './src/index';
import { numberAsc, byField, compose } from '../公共库/src/比较器';

// 近乎有序的数据
const nearlySorted = [1, 2, 3, 5, 4, 6, 7, 9, 8, 10];
const sorted = sort(nearlySorted, numberAsc);
// 接近 O(n) 的性能！

// 表格多列稳定排序
const employees = [...];
// 第一次：按部门排序
const byDept = sort(employees, (a, b) => a.dept.localeCompare(b.dept));
// 第二次：按薪资排序（保持部门顺序）
const final = sort(byDept, (a, b) => b.salary - a.salary);
```

## 🔧 API

### `sort<T>(arr, cmp): T[]`
返回排序后的新数组。

### `sortInPlace<T>(arr, cmp): T[]`
原地排序。

### `meta`
算法元信息。

## 📊 复杂度

| 指标 | 值 | 说明 |
|------|-----|------|
| 时间（最好） | **O(n)** ⭐ | 完全有序时 |
| 时间（平均/最坏） | O(n log n) | |
| 空间 | O(n) | 合并需要 |
| 稳定性 | ✅ **稳定** | 核心优势 |

## 🚨 何时使用

- ✅ 数据近乎有序
- ✅ 需要稳定排序
- ✅ 表格多列排序
- ❌ 完全随机数据（Introsort 更好）

## 📁 文件结构

```
TimSort/
├── README.md
├── src/
│   ├── index.ts    # 核心实现
│   └── demo.ts     # 使用示例
└── test/
    └── index.test.ts
```

