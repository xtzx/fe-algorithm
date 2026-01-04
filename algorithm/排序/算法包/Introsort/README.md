# Introsort (内省排序)

> 快排 + 堆排 + 插入的完美结合：工业级通用排序

## 📦 快速使用

```typescript
import { sort, sortInPlace, meta } from './src/index';
import { numberAsc } from '../公共库/src/比较器';

// 通用排序
const numbers = [38, 27, 43, 3, 9, 82, 10];
const sorted = sort(numbers, numberAsc);

// 不怕对抗性输入
const adversarial = generateAdversarialData();
const safe = sort(adversarial, numberAsc); // 保证 O(n log n)
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
| 时间（所有情况） | O(n log n) | ⭐ 堆排兜底 |
| 空间 | O(log n) | 递归栈 |
| 稳定性 | ❌ | 不稳定 |

## 🔄 决策流程

```
开始 → 数组小? → 插入排序
         ↓ No
     深度超限? → 堆排序
         ↓ No
     快排 partition → 递归两边
```

## 📁 文件结构

```
Introsort/
├── README.md
├── src/
│   ├── index.ts    # 核心实现
│   └── demo.ts     # 使用示例
└── test/
    └── index.test.ts
```

