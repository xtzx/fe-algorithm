# 冒泡排序 (Bubble Sort)

> 经典入门排序算法，通过相邻元素比较交换，让大元素"冒泡"到末尾。

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
//   name: '冒泡排序',
//   stable: true,
//   inPlace: true,
//   timeComplexity: { best: 'O(n)', average: 'O(n²)', worst: 'O(n²)' },
//   spaceComplexity: 'O(1)',
//   适用场景: ['教学演示', '小规模数据', '检测数组是否有序'],
//   不适用场景: ['大规模数据', '性能敏感场景']
// }
```

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

### 对象数组排序

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

// 按分数降序
const sorted = sort(students, (a, b) => b.score - a.score);
// [{ name: 'Bob', score: 90 }, { name: 'Alice', score: 85 }, { name: 'Charlie', score: 85 }]

// 稳定性：Alice 和 Charlie 分数相同，保持原始顺序
```

### 多列排序（利用稳定性）

```typescript
// 先按次要列排序
let data = sort(employees, (a, b) => a.name.localeCompare(b.name));
// 再按主要列排序
data = sort(data, (a, b) => b.score - a.score);
// 同分数的员工保持姓名排序顺序
```

---

## 🎯 适用场景

| 场景 | 推荐 | 原因 |
|------|:----:|------|
| 教学演示 | ✅ | 逻辑简单易懂 |
| 小规模数据 (<50) | ✅ | 常数因子小 |
| 需要稳定排序 | ✅ | 相等元素保持顺序 |
| 检测是否有序 | ✅ | 优化版一轮无交换即结束 |
| 大规模数据 | ❌ | O(n²) 太慢 |
| 性能敏感场景 | ❌ | 交换次数多 |

---

## 📊 复杂度

| 指标 | 复杂度 |
|------|--------|
| 时间（最好） | O(n) |
| 时间（平均） | O(n²) |
| 时间（最坏） | O(n²) |
| 空间 | O(1) |
| 稳定性 | ✅ 稳定 |

---

## 📖 详细原理

请参阅 [文档/算法详解/比较类排序/01-冒泡排序.md](../../文档/算法详解/比较类排序/01-冒泡排序.md)
