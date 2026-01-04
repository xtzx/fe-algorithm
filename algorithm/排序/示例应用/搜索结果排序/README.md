# 搜索结果排序示例应用

## 📌 功能概述

演示搜索结果排序的常见需求：
- TopK 高效获取最相关的 K 条结果
- 增量更新（流式数据）
- 分页与游标

## 📁 文件结构

```
搜索结果排序/
├── README.md
├── src/
│   ├── 数据模型.ts      # 搜索结果类型定义
│   ├── TopK小顶堆.ts    # TopK 堆实现
│   ├── 增量更新排序.ts   # 流式数据 TopK 追踪
│   ├── 分页与游标.ts    # 分页工具
│   └── demo.ts          # 综合示例
└── test/
    └── index.test.ts    # 测试用例
```

## 🚀 快速使用

### TopK 堆

```typescript
import { topKByHeap } from './src/TopK小顶堆';

const topResults = topKByHeap(
  allResults,
  20,
  (a, b) => b.relevance - a.relevance
);
```

### 增量更新

```typescript
import { TopKTracker } from './src/增量更新排序';

const tracker = new TopKTracker(10, (a, b) => a.relevance - b.relevance);

// 流式添加
for (const result of stream) {
  tracker.add(result);
}

const top10 = tracker.getTopK();
```

### 游标分页

```typescript
import { paginateWithCursor } from './src/分页与游标';

const page = paginateWithCursor(
  sortedResults,
  cursor,
  20,
  r => r.id
);
```

## 📊 TopK 方案对比

| 方法 | 时间复杂度 | 适用场景 |
|------|-----------|---------|
| 全量排序 | O(n log n) | k ≈ n |
| TopK 堆 | O(n log k) | k << n，流式数据 |
| 快速选择 | O(n) 平均 | 一次性处理 |

