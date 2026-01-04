# 并行归并排序

> 利用 Web Worker 实现多线程并行排序，加速大规模数据处理

## 📖 算法简介

将大数组分成多个块，每个块在独立的 Worker 中排序，最后通过 K 路归并合并结果。

## 🚀 使用方法

```typescript
import { parallelMergeSort, shouldUseParallel } from './src';

// 基础用法
const numbers = Array.from({ length: 100000 }, () => Math.random());
const sorted = await parallelMergeSort(numbers, (a, b) => a - b);

// 指定 Worker 数量
const sorted = await parallelMergeSort(numbers, (a, b) => a - b, {
  workerCount: 4,
});

// 判断是否值得并行
if (shouldUseParallel(data.length)) {
  // 使用并行排序
}
```

## ⚙️ 配置选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `workerCount` | number | `navigator.hardwareConcurrency` | Worker 数量 |
| `threshold` | number | 10000 | 低于此值使用单线程 |

## 📊 性能参考

| 数据量 | 单线程 | 2 Workers | 4 Workers |
|-------:|-------:|----------:|----------:|
| 10,000 | 5ms | ~8ms | ~12ms |
| 100,000 | 55ms | ~35ms | ~28ms |
| 1,000,000 | 650ms | ~350ms | ~220ms |

## 📁 目录结构

```
并行归并排序/
├── README.md
├── src/
│   ├── index.ts      # 主线程调度
│   └── worker.ts     # Worker 排序逻辑
└── test/
    └── index.test.ts # 测试文件
```

## ⚠️ 注意事项

1. **数据量阈值**：数据量太小时，Worker 创建和通信开销会超过并行收益
2. **对象排序**：复杂对象的序列化开销较大，建议减少 Worker 数量
3. **比较函数**：需要能被序列化到 Worker（不能捕获外部变量）
4. **浏览器支持**：需要支持 Web Worker 的现代浏览器

## 🔗 相关链接

- [Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
- [Transferable Objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects)

