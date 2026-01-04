# 希尔排序 (Shell Sort)

> 插入排序的优化版：通过**分组**进行插入排序，逐步缩小间隔，让元素更快移动到目标位置。

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

### `sortWithGaps<T>(arr, cmp, gaps): T[]`

使用自定义间隔序列排序。

```typescript
// 使用 Hibbard 序列
sortWithGaps(arr, cmp, [15, 7, 3, 1]);
```

### `meta`

算法元信息。

```typescript
console.log(meta);
// {
//   name: '希尔排序',
//   stable: false,    // ⚠️ 不稳定
//   inPlace: true,
//   timeComplexity: { best: 'O(n log n)', average: 'O(n^1.3)', worst: 'O(n²)' },
//   spaceComplexity: 'O(1)',
//   适用场景: ['中等规模数据', '不需要稳定性'],
//   不适用场景: ['需要稳定排序']
// }
```

---

## ⚠️ 重要提醒：不稳定排序

希尔排序是**不稳定**的排序算法！相等元素的相对顺序可能被改变。

如果需要稳定排序，请使用冒泡排序、插入排序或 TimSort。

---

## 📝 使用示例

### 基础排序

```typescript
// 数字升序
sort([5, 3, 8], (a, b) => a - b);  // [3, 5, 8]

// 数字降序
sort([5, 3, 8], (a, b) => b - a);  // [8, 5, 3]
```

### 中等规模数据

希尔排序在中等规模数据（100-10000）上表现良好：

```typescript
const mediumData = generateNumbers(5000, 'random');
const sorted = sort(mediumData, (a, b) => a - b);
// 比 O(n²) 算法快很多，实现又比快排简单
```

---

## 🎯 适用场景

| 场景 | 推荐 | 原因 |
|------|:----:|------|
| 中等规模数据 | ✅ | 比 O(n²) 快，实现简单 |
| 不需要稳定性 | ✅ | |
| 内存受限 | ✅ | O(1) 空间 |
| 需要稳定排序 | ❌ | **不稳定** |
| 追求极致性能 | ❌ | 快排更好 |

---

## 📊 复杂度

| 指标 | 复杂度 |
|------|--------|
| 时间（最好） | O(n log n) |
| 时间（平均） | O(n^1.3) ~ O(n^1.5) |
| 时间（最坏） | O(n²) |
| 空间 | O(1) |
| 稳定性 | ❌ 不稳定 |

---

## 🔧 间隔序列

不同的间隔序列影响性能：

| 序列 | 最坏复杂度 | 说明 |
|------|-----------|------|
| Shell (n/2) | O(n²) | 原始序列 |
| Hibbard | O(n^1.5) | 2^k - 1 |
| **Knuth** | O(n^1.5) | (3^k - 1)/2，推荐 |
| Sedgewick | O(n^4/3) | 更好但复杂 |

默认使用 **Knuth 序列**。

---

## 📖 详细原理

请参阅 [文档/算法详解/比较类排序/04-希尔排序.md](../../文档/算法详解/比较类排序/04-希尔排序.md)
