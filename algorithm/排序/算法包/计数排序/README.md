# 计数排序 (Counting Sort)

## 📌 适用条件

- ✅ **整数**（或可离散化为整数）
- ✅ **值域较小**（k 不远大于 n）
- ✅ 需要**稳定排序**

## 📥 导入

```typescript
import {
  countingSort,
  countingSortAuto,
  countingSortBy,
  meta,
} from './src/index';
```

## 🚀 快速使用

### 数字排序

```typescript
// 指定范围
const sorted = countingSort([4, 2, 8, 5, 2, 3], 0, 10);

// 自动检测范围
const sorted = countingSortAuto([4, 2, 8, 5, 2, 3]);
```

### 对象排序

```typescript
interface Student {
  name: string;
  score: number; // 0-100
}

const students: Student[] = [
  { name: 'Alice', score: 85 },
  { name: 'Bob', score: 92 },
  { name: 'Charlie', score: 85 },
];

// 按分数排序（稳定）
const sorted = countingSortBy(students, s => s.score, 0, 100);
// 同分的 Alice 和 Charlie 保持原顺序
```

## 📊 复杂度

| 指标 | 值 |
|------|-----|
| 时间 | O(n + k) |
| 空间 | O(n + k) |
| 稳定性 | ✅ 稳定 |

> k 是值域大小（max - min + 1）

## ⚠️ 注意事项

1. 需要指定值域范围，超出范围会抛出错误
2. 不适用于值域极大的场景（空间浪费）
3. 不适用于浮点数（除非离散化）

