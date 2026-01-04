# 桶排序 (Bucket Sort)

## 📌 适用条件

- ✅ 数据**分布均匀**
- ✅ 能定义合理的**桶映射函数**
- ✅ 桶数量合理（通常 ≈ n）

## 📥 导入

```typescript
import {
  bucketSort,
  bucketSortGeneric,
  bucketSortStable,
  meta,
} from './src/index';
```

## 🚀 快速使用

### 浮点数排序（0-1 范围）

```typescript
const arr = [0.78, 0.17, 0.39, 0.26, 0.72];
const sorted = bucketSort(arr);
```

### 通用桶排序

```typescript
interface Product {
  name: string;
  price: number; // 0-1000
}

const products: Product[] = [...];

// 每 50 元一个桶
const bucketCount = 20;
const getBucket = (p: Product) => Math.min(19, Math.floor(p.price / 50));
const cmp = (a: Product, b: Product) => a.price - b.price;

const sorted = bucketSortGeneric(products, bucketCount, getBucket, cmp);
```

### 稳定版本

```typescript
// 使用插入排序保证稳定性
const sorted = bucketSortStable(products, bucketCount, getBucket, cmp);
```

## 📊 复杂度

| 指标 | 均匀分布 | 最坏情况 |
|------|---------|---------|
| 时间 | O(n) | O(n²) |
| 空间 | O(n + k) | O(n + k) |
| 稳定性 | 取决于桶内排序 | - |

> 分布不均匀时性能下降

## ⚠️ 注意事项

1. 需要设计合适的映射函数
2. 数据分布极不均匀时不适用
3. 稳定性取决于桶内使用的排序算法

