# 外部归并排序 (External Merge Sort)

## 📌 适用条件

- ✅ 数据**无法一次装入内存**
- ✅ 需要对**大文件/大数组**排序
- ✅ 流式数据处理
- ✅ 前端场景：大数组分片处理、Web Worker 并行排序

## 📥 导入

```typescript
import {
  externalMergeSort,
  kWayMerge,
  createExternalSorter,
  StreamSorter,
  meta,
} from './src/index';
```

## 🚀 快速使用

### 基础用法

```typescript
import { numberAsc } from '../../公共库/src/比较器';

const largeArray = [...]; // 假设很大的数组
const chunkSize = 10000;   // 每块 1 万个元素

const sorted = externalMergeSort(largeArray, chunkSize, numberAsc);
```

### 多路归并

```typescript
// 已经有多个有序数组，需要合并
const sortedChunks = [
  [1, 4, 7],
  [2, 5, 8],
  [3, 6, 9],
];

const merged = kWayMerge(sortedChunks, numberAsc);
// [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 可配置版本

```typescript
import { createExternalSorter } from './src/index';
import { timSort } from '../TimSort/src/index';

const sorter = createExternalSorter({
  chunkSize: 10000,
  sortChunk: (chunk, cmp) => timSort(chunk, cmp),
  mergeChunks: kWayMerge,
});

const sorted = sorter(largeArray, numberAsc);
```

### 流式处理

```typescript
import { StreamSorter } from './src/index';

const sorter = new StreamSorter<number>(1000, numberAsc);

// 流式添加数据
for (const item of dataStream) {
  sorter.add(item);
}

// 获取最终结果
const sorted = sorter.getResult();
```

## 📊 复杂度

| 指标 | 值 |
|------|-----|
| 时间 | O(n log n) |
| 空间 | O(n) |
| I/O | O(n/B · log_{M/B}(n/B)) |

> B 是块大小，M 是可用内存

## 🔧 设计特点

- 可插拔的 `sortChunk`（默认使用原生 sort）
- 可插拔的 `mergeChunks`（默认使用最小堆 K 路归并）
- 流式处理器 `StreamSorter`

## ⚠️ 注意事项

1. 这是内存模拟版本，真正的外部排序需要文件 I/O
2. 前端可用于分片处理大数组
3. 配合 Web Worker 可实现并行排序

