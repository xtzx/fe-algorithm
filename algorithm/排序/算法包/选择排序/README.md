# 选择排序 (Selection Sort)

> 每轮从未排序部分**选择最小元素**，放到已排序部分的末尾。

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
//   name: '选择排序',
//   stable: false,    // ⚠️ 不稳定
//   inPlace: true,
//   timeComplexity: { best: 'O(n²)', average: 'O(n²)', worst: 'O(n²)' },
//   spaceComplexity: 'O(1)',
//   适用场景: ['交换成本高', '小规模数据'],
//   不适用场景: ['需要稳定排序', '大规模数据']
// }
```

---

## ⚠️ 重要提醒：不稳定排序

选择排序是**不稳定**的排序算法！相等元素的相对顺序可能被改变。

```typescript
const data = [
  { key: 3, id: 'a' },
  { key: 1, id: 'b' },
  { key: 3, id: 'c' },
];

// 排序后，两个 key=3 的元素顺序可能变为 c, a
// 如果需要稳定排序，请使用冒泡排序或插入排序
```

---

## 📝 使用示例

### 基础排序

```typescript
// 数字升序
sort([5, 3, 8], (a, b) => a - b);  // [3, 5, 8]

// 数字降序
sort([5, 3, 8], (a, b) => b - a);  // [8, 5, 3]
```

### 交换成本高的场景

选择排序只需 O(n) 次交换，适合交换操作昂贵的场景。

```typescript
// 例如：需要移动大型对象或 DOM 元素
const heavyObjects = [...];
sortInPlace(heavyObjects, compareFunc);
// 最多 n-1 次交换，比冒泡排序的 O(n²) 次交换少很多
```

---

## 🎯 适用场景

| 场景 | 推荐 | 原因 |
|------|:----:|------|
| 交换成本高 | ✅ | 只需 O(n) 次交换 |
| 小规模数据 | ✅ | 实现简单 |
| 需要稳定排序 | ❌ | **不稳定** |
| 大规模数据 | ❌ | O(n²) 太慢 |

---

## 📊 复杂度

| 指标 | 复杂度 |
|------|--------|
| 时间（最好） | O(n²) |
| 时间（平均） | O(n²) |
| 时间（最坏） | O(n²) |
| 空间 | O(1) |
| 稳定性 | ❌ 不稳定 |

---

## 📖 详细原理

请参阅 [文档/算法详解/比较类排序/02-选择排序.md](../../文档/算法详解/比较类排序/02-选择排序.md)
