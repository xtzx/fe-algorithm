# 插入排序 (Insertion Sort)

> 像整理扑克牌一样，将每张新牌**插入**到已排序部分的正确位置。

---

## 📦 安装与使用

```typescript
import { sort, sortInPlace, meta } from './src';
```

---

## 🚀 API

### `sort<T>(arr, cmp): T[]`

创建新数组进行排序，**不修改原数组**。

```typescript
const numbers = [5, 3, 8, 4, 2];
const sorted = sort(numbers, (a, b) => a - b);

console.log(sorted);   // [2, 3, 4, 5, 8]
console.log(numbers);  // [5, 3, 8, 4, 2] 原数组不变
```

### `sortInPlace<T>(arr, cmp): T[]`

**原地排序**，修改并返回原数组。

```typescript
const numbers = [5, 3, 8, 4, 2];
sortInPlace(numbers, (a, b) => a - b);

console.log(numbers);  // [2, 3, 4, 5, 8]
```

### `meta`

算法元信息。

```typescript
console.log(meta);
// {
//   name: '插入排序',
//   stable: true,
//   inPlace: true,
//   timeComplexity: { best: 'O(n)', average: 'O(n²)', worst: 'O(n²)' },
//   spaceComplexity: 'O(1)',
//   适用场景: ['小规模数据', '近乎有序数据', '在线排序'],
//   不适用场景: ['大规模随机数据']
// }
```

---

## ⭐ 核心优势：近乎有序数据极快

当数据**基本有序**时，插入排序的时间复杂度接近 **O(n)**！

```typescript
// 近乎有序的数据
const nearlySorted = [1, 2, 4, 3, 5, 6, 8, 7, 9, 10];

// 插入排序在这种情况下非常快
const sorted = sort(nearlySorted, (a, b) => a - b);
```

这也是为什么 **TimSort** 和 **Introsort** 都使用插入排序处理小数组。

---

## 📝 使用示例

### 基础排序

```typescript
// 数字升序
sort([5, 3, 8], (a, b) => a - b);  // [3, 5, 8]

// 数字降序
sort([5, 3, 8], (a, b) => b - a);  // [8, 5, 3]

// 字符串排序
sort(['c', 'a', 'b'], (a, b) => a.localeCompare(b));  // ['a', 'b', 'c']
```

### 对象数组排序（稳定）

```typescript
interface Student {
  name: string;
  score: number;
}

const students: Student[] = [
  { name: 'Alice', score: 85 },
  { name: 'Bob', score: 90 },
  { name: 'Charlie', score: 85 },
];

// 按分数降序，稳定排序保持同分学生的原始顺序
const sorted = sort(students, (a, b) => b.score - a.score);
// Alice 和 Charlie 分数相同，Alice 仍在前
```

### 在线排序（增量插入）

```typescript
import { insertSorted } from './src';

// 维护一个有序列表，新元素可以高效插入
const sortedList = [1, 3, 5, 7];
insertSorted(sortedList, 4, (a, b) => a - b);
// [1, 3, 4, 5, 7]
```

---

## 🎯 适用场景

| 场景 | 推荐 | 原因 |
|------|:----:|------|
| 小规模数据 (<50) | ✅ | 常数因子小 |
| **近乎有序数据** | ✅⭐ | O(n) 时间复杂度 |
| 在线排序（增量） | ✅ | 单次插入 O(n) |
| 需要稳定排序 | ✅ | 相等元素保持顺序 |
| 作为其他算法子程序 | ✅ | TimSort/Introsort 都用 |
| 大规模随机数据 | ❌ | O(n²) 太慢 |

---

## 📊 复杂度

| 指标 | 复杂度 |
|------|--------|
| 时间（最好） | **O(n)** ⭐ |
| 时间（平均） | O(n²) |
| 时间（最坏） | O(n²) |
| 空间 | O(1) |
| 稳定性 | ✅ 稳定 |

---

## 📖 详细原理

请参阅 [文档/算法详解/比较类排序/03-插入排序.md](../../文档/算法详解/比较类排序/03-插入排序.md)
