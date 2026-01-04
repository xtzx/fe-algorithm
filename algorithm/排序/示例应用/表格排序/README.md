# 表格排序示例应用

## 📌 功能概述

演示前端表格多列排序的实现方式：
- 多列稳定排序
- 比较器组合
- 动态列排序

## 📁 文件结构

```
表格排序/
├── README.md
├── src/
│   ├── 数据模型.ts      # 表格数据类型定义
│   ├── 多列稳定排序.ts   # 多次排序实现
│   ├── 比较器组合器.ts   # 组合比较器实现
│   └── demo.ts          # 综合示例
└── test/
    └── index.test.ts    # 测试用例
```

## 🚀 快速使用

### 多列稳定排序

```typescript
import { sortByMultipleColumns } from './src/多列稳定排序';

const sorted = sortByMultipleColumns(data, [
  { field: 'department', order: 'asc' },
  { field: 'score', order: 'desc' },
]);
```

### 比较器组合

```typescript
import { createTableComparator } from './src/比较器组合器';

const comparator = createTableComparator([
  { field: 'department', order: 'asc', type: 'string' },
  { field: 'score', order: 'desc', type: 'number' },
]);

const sorted = [...data].sort(comparator);
```

## 📊 两种方式对比

| 维度 | 多列稳定排序 | 比较器组合 |
|------|------------|-----------|
| 性能 | O(k · n log n) | O(n log n) |
| 可读性 | 顺序与优先级相反 | 顺序与优先级一致 |
| 推荐 | 少量列 | 性能敏感 |

