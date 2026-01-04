/**
 * 外部归并排序示例
 *
 * 演示外部归并排序在前端场景中的应用
 */

import {
  externalMergeSort,
  kWayMerge,
  twoWayMerge,
  iterativeMerge,
  createExternalSorter,
  StreamSorter,
} from './index';
import { numberAsc, byField } from '../../公共库/src/比较器';

// ============================================================================
// 示例 1：基础用法
// ============================================================================

console.log('=== 示例 1：基础外部归并排序 ===');

const largeArray = Array.from({ length: 100 }, () =>
  Math.floor(Math.random() * 1000)
);

console.log('原数组（前10个）:', largeArray.slice(0, 10));

// 模拟内存限制：每块最多 20 个元素
const sorted = externalMergeSort(largeArray, 20, numberAsc);

console.log('排序后（前10个）:', sorted.slice(0, 10));
console.log('分成了', Math.ceil(largeArray.length / 20), '个块进行排序');

// ============================================================================
// 示例 2：K 路归并
// ============================================================================

console.log('\n=== 示例 2：K 路归并 ===');

const sortedChunks = [
  [1, 4, 7, 10],
  [2, 5, 8, 11],
  [3, 6, 9, 12],
];

console.log('有序块:');
sortedChunks.forEach((chunk, i) =>
  console.log(`  块${i + 1}: [${chunk.join(', ')}]`)
);

const merged = kWayMerge(sortedChunks, numberAsc);
console.log('合并结果:', merged);

// ============================================================================
// 示例 3：表格数据排序
// ============================================================================

console.log('\n=== 示例 3：大表格数据排序 ===');

interface TableRow {
  id: number;
  name: string;
  score: number;
  timestamp: number;
}

// 模拟大量表格数据
const tableData: TableRow[] = Array.from({ length: 50 }, (_, i) => ({
  id: i + 1,
  name: `User${String(i + 1).padStart(3, '0')}`,
  score: Math.floor(Math.random() * 100),
  timestamp: Date.now() - Math.floor(Math.random() * 86400000),
}));

// 按分数排序
const sortedByScore = externalMergeSort(
  tableData,
  10, // 每块 10 条
  byField('score', numberAsc)
);

console.log('按分数排序（前5条）:');
console.log('  ID   | Name    | Score');
console.log('  -----|---------|------');
sortedByScore.slice(0, 5).forEach(r =>
  console.log(`  ${String(r.id).padStart(4)} | ${r.name} | ${r.score}`)
);

// ============================================================================
// 示例 4：流式数据处理
// ============================================================================

console.log('\n=== 示例 4：流式数据处理 ===');

const streamSorter = new StreamSorter<number>(10, numberAsc);

// 模拟流式数据到达
console.log('模拟流式数据到达...');

const batch1 = [45, 23, 78, 12, 56];
console.log('  批次1:', batch1);
streamSorter.addBatch(batch1);

const batch2 = [89, 34, 67, 90, 11];
console.log('  批次2:', batch2);
streamSorter.addBatch(batch2);

const batch3 = [42, 88, 15, 73, 29];
console.log('  批次3:', batch3);
streamSorter.addBatch(batch3);

console.log('当前块数量:', streamSorter.getChunkCount());

const streamResult = streamSorter.getResult();
console.log('最终排序结果:', streamResult);

// ============================================================================
// 示例 5：搜索结果排序
// ============================================================================

console.log('\n=== 示例 5：搜索结果排序 ===');

interface SearchResult {
  id: string;
  title: string;
  relevance: number;
  viewCount: number;
}

// 模拟搜索结果
const searchResults: SearchResult[] = Array.from({ length: 30 }, (_, i) => ({
  id: `result-${i + 1}`,
  title: `Article ${i + 1}`,
  relevance: Math.random() * 100,
  viewCount: Math.floor(Math.random() * 10000),
}));

// 按相关度降序排序
const byRelevanceDesc = externalMergeSort(
  searchResults,
  10,
  (a, b) => b.relevance - a.relevance
);

console.log('按相关度排序（前5条）:');
byRelevanceDesc.slice(0, 5).forEach(r =>
  console.log(`  ${r.title}: 相关度 ${r.relevance.toFixed(2)}`)
);

// ============================================================================
// 示例 6：合并多个数据源
// ============================================================================

console.log('\n=== 示例 6：合并多个有序数据源 ===');

// 模拟从多个 API 获取的有序数据
const source1 = [100, 200, 300, 400, 500].map(ts => ({
  source: 'API1',
  timestamp: ts,
}));

const source2 = [150, 250, 350, 450].map(ts => ({
  source: 'API2',
  timestamp: ts,
}));

const source3 = [120, 220, 320].map(ts => ({
  source: 'API3',
  timestamp: ts,
}));

console.log('数据源1:', source1.map(s => s.timestamp));
console.log('数据源2:', source2.map(s => s.timestamp));
console.log('数据源3:', source3.map(s => s.timestamp));

const mergedSources = kWayMerge(
  [source1, source2, source3],
  (a, b) => a.timestamp - b.timestamp
);

console.log('合并后（按时间戳）:');
mergedSources.forEach(s =>
  console.log(`  [${s.source}] ${s.timestamp}`)
);

// ============================================================================
// 示例 7：可配置的排序器
// ============================================================================

console.log('\n=== 示例 7：可配置的排序器 ===');

// 自定义排序配置
const customSorter = createExternalSorter<number>({
  chunkSize: 15,
  sortChunk: (chunk, cmp) => {
    console.log(`  排序块，大小: ${chunk.length}`);
    return chunk.sort(cmp);
  },
  mergeChunks: (chunks, cmp) => {
    console.log(`  合并 ${chunks.length} 个块`);
    return kWayMerge(chunks, cmp);
  },
});

const customData = Array.from({ length: 50 }, () =>
  Math.floor(Math.random() * 100)
);

console.log('使用自定义排序器:');
const customResult = customSorter(customData, numberAsc);
console.log('结果（前10个）:', customResult.slice(0, 10));

// ============================================================================
// 示例 8：两路归并 vs K 路归并
// ============================================================================

console.log('\n=== 示例 8：两路归并 vs K 路归并 ===');

const chunk1 = [1, 3, 5, 7, 9];
const chunk2 = [2, 4, 6, 8, 10];

console.log('块1:', chunk1);
console.log('块2:', chunk2);

// 两路归并
const twoWayResult = twoWayMerge(chunk1, chunk2, numberAsc);
console.log('两路归并:', twoWayResult);

// 迭代归并（多块）
const multiChunks = [[1, 5, 9], [2, 6], [3, 7], [4, 8, 10]];
console.log('\n多块:', multiChunks);

const iterativeResult = iterativeMerge(multiChunks, numberAsc);
console.log('迭代归并:', iterativeResult);

// ============================================================================
// 示例 9：讨论使用场景
// ============================================================================

console.log('\n=== 示例 9：使用场景讨论 ===');

interface UseCase {
  scenario: string;
  dataSize: string;
  recommendation: string;
  reason: string;
}

const useCases: UseCase[] = [
  {
    scenario: '前端大数组分片',
    dataSize: '> 10万条',
    recommendation: '✅ 推荐',
    reason: '避免阻塞主线程',
  },
  {
    scenario: 'Web Worker 并行排序',
    dataSize: '> 100万条',
    recommendation: '✅ 强烈推荐',
    reason: '分块并行处理',
  },
  {
    scenario: '合并多个有序数据源',
    dataSize: '任意',
    recommendation: '✅ 推荐',
    reason: 'K 路归并最优',
  },
  {
    scenario: '流式数据排序',
    dataSize: '不定',
    recommendation: '✅ 推荐',
    reason: 'StreamSorter 适合',
  },
  {
    scenario: '小数组 (< 1000)',
    dataSize: '< 1000条',
    recommendation: '❌ 不需要',
    reason: '直接排序更简单',
  },
];

console.log('场景分析:');
console.log('场景               | 数据量     | 推荐   | 原因');
console.log('-------------------|------------|--------|--------------------');
useCases.forEach(u =>
  console.log(
    `${u.scenario.padEnd(19)}| ${u.dataSize.padEnd(10)} | ${u.recommendation} | ${u.reason}`
  )
);

// ============================================================================
// 示例 10：性能提示
// ============================================================================

console.log('\n=== 示例 10：性能提示 ===');

console.log(`
性能优化建议:

1. 块大小选择
   - 太小：归并开销大
   - 太大：可能超出内存限制
   - 推荐：根据可用内存和数据类型调整

2. 归并策略
   - K 路归并：适合块数较多的情况
   - 两路迭代：实现简单，适合块数较少

3. 并行化
   - 块内排序可以并行（Web Worker）
   - 归并阶段需要同步

4. 前端特殊考虑
   - 使用 requestIdleCallback 分片处理
   - 使用 Web Worker 避免阻塞 UI
   - 考虑内存占用，及时释放中间结果
`);

console.log('✅ 所有示例运行完成');

