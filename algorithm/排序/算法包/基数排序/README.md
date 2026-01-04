# 基数排序 (Radix Sort)

## 📌 适用条件

- ✅ **整数**或**定长字符串**
- ✅ 位数有限（d 不太大）
- ✅ 需要**稳定排序**
- ✅ 默认**非负整数**（负数需特殊处理）

## 📥 导入

```typescript
import {
  radixSort,
  radixSortBy,
  radixSortWithNegative,
  meta,
} from './src/index';
```

## 🚀 快速使用

### 数字排序

```typescript
// 非负整数
const sorted = radixSort([170, 45, 75, 90, 802, 24, 2, 66]);

// 支持负数
const sorted = radixSortWithNegative([-5, 3, -2, 8, -1]);
```

### 对象排序

```typescript
interface Order {
  id: number; // 8位订单号
  amount: number;
}

const orders: Order[] = [...];

// 按订单号排序
const sorted = radixSortBy(orders, o => o.id);
```

## 📊 复杂度

| 指标 | 值 |
|------|-----|
| 时间 | O(d · (n + k)) |
| 空间 | O(n + k) |
| 稳定性 | ✅ 稳定 |

> d 是位数，k 是基数（默认 10）

## 🔧 变种

| 类型 | 方向 | 适用场景 |
|------|------|----------|
| LSD | 低位 → 高位 | 等长数据（默认） |
| MSD | 高位 → 低位 | 变长数据、可提前终止 |

## ⚠️ 注意事项

1. 默认只支持非负整数
2. 负数需要使用 `radixSortWithNegative`
3. 不适用于浮点数

