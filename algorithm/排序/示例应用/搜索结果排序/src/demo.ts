/**
 * 搜索结果排序综合示例
 */

import {
  generateSearchResults,
  generateWithRelevanceDistribution,
  printResults,
  type SearchResult,
} from './数据模型';
import {
  topKByHeap,
  topKBySort,
  topKWithFilter,
  topKMultiDimension,
} from './TopK小顶堆';
import {
  TopKTracker,
  SlidingWindowTopK,
  Leaderboard,
} from './增量更新排序';
import {
  paginateByOffset,
  paginateWithCursor,
  PaginationManager,
} from './分页与游标';
import { numberAsc } from '../../../算法包/公共库/src/比较器';

// ============================================================================
// 示例 1：TopK 基础用法
// ============================================================================

console.log('=== 示例 1：TopK 基础用法 ===\n');

const results = generateSearchResults(1000);

// 获取相关度最高的 10 条
const top10 = topKByHeap(results, 10, (a, b) => a.relevance - b.relevance);

console.log('相关度 Top 10:');
printResults(top10, ['id', 'title', 'relevance']);

// ============================================================================
// 示例 2：TopK 方案对比
// ============================================================================

console.log('\n=== 示例 2：TopK 方案对比 ===\n');

const largeData = generateSearchResults(100000);
const k = 20;
const cmp = (a: SearchResult, b: SearchResult) => a.relevance - b.relevance;

// 方案 1：全量排序
const start1 = performance.now();
const result1 = topKBySort(largeData, k, cmp);
const duration1 = performance.now() - start1;

// 方案 2：TopK 堆
const start2 = performance.now();
const result2 = topKByHeap(largeData, k, cmp);
const duration2 = performance.now() - start2;

console.log(`数据量: ${largeData.length}, K = ${k}`);
console.log(`全量排序: ${duration1.toFixed(2)} ms`);
console.log(`TopK 堆:  ${duration2.toFixed(2)} ms`);
console.log(`性能提升: ${((duration1 - duration2) / duration1 * 100).toFixed(1)}%`);

// 验证结果一致性
const ids1 = result1.map(r => r.id).join(',');
const ids2 = result2.map(r => r.id).join(',');
console.log(`结果一致: ${ids1 === ids2 ? '✅' : '❌'}`);

// ============================================================================
// 示例 3：带过滤的 TopK
// ============================================================================

console.log('\n=== 示例 3：带过滤的 TopK ===\n');

const now = Date.now();
const oneWeek = 7 * 24 * 60 * 60 * 1000;

// 只要最近一周的热门结果
const recentTop = topKWithFilter(
  results,
  5,
  r => r.publishTime > now - oneWeek,
  (a, b) => a.viewCount - b.viewCount
);

console.log('最近一周浏览量 Top 5:');
printResults(recentTop, ['id', 'title', 'viewCount']);

// ============================================================================
// 示例 4：多维度排序
// ============================================================================

console.log('\n=== 示例 4：多维度排序 ===\n');

const multiDimTop = topKMultiDimension(results, 5, [
  { getValue: r => r.relevance, weight: 0.5, order: 'desc' },
  { getValue: r => r.viewCount, weight: 0.3, order: 'desc' },
  { getValue: r => r.publishTime, weight: 0.2, order: 'desc' },
]);

console.log('综合评分 Top 5（相关度50% + 浏览量30% + 时间20%）:');
printResults(multiDimTop, ['id', 'relevance', 'viewCount']);

// ============================================================================
// 示例 5：增量更新（流式数据）
// ============================================================================

console.log('\n=== 示例 5：增量更新（流式数据）===\n');

const tracker = new TopKTracker<SearchResult>(5, (a, b) => a.relevance - b.relevance);

// 模拟流式数据到达
const batch1 = generateSearchResults(10).map((r, i) => ({ ...r, id: `batch1-${i}` }));
console.log('批次 1 到达（10 条）');
tracker.addBatch(batch1);
console.log('当前 Top 5:');
printResults(tracker.getTopK(), ['id', 'relevance']);

const batch2 = generateWithRelevanceDistribution(10, 'biased')
  .map((r, i) => ({ ...r, id: `batch2-${i}`, relevance: 80 + Math.floor(Math.random() * 20) }));
console.log('\n批次 2 到达（10 条高分）');
const entered = tracker.addBatch(batch2);
console.log(`进入 Top 5 的数量: ${entered}`);
console.log('更新后 Top 5:');
printResults(tracker.getTopK(), ['id', 'relevance']);

console.log('\n统计信息:', tracker.getStats());

// ============================================================================
// 示例 6：滑动窗口 TopK
// ============================================================================

console.log('\n=== 示例 6：滑动窗口 TopK ===\n');

const slidingTopK = new SlidingWindowTopK<SearchResult>(
  100, // 窗口大小
  3,   // Top 3
  (a, b) => a.relevance - b.relevance
);

// 模拟持续到达的数据
for (let i = 0; i < 150; i++) {
  slidingTopK.add({
    id: `sliding-${i}`,
    title: `Result ${i}`,
    description: '',
    relevance: Math.floor(Math.random() * 100),
    publishTime: Date.now(),
    viewCount: 0,
    likeCount: 0,
    category: '',
  });
}

console.log(`窗口大小: ${slidingTopK.size()}`);
console.log('最近 100 条中的 Top 3:');
printResults(slidingTopK.getTopK(), ['id', 'relevance']);

// ============================================================================
// 示例 7：实时排行榜
// ============================================================================

console.log('\n=== 示例 7：实时排行榜 ===\n');

interface Player {
  id: string;
  name: string;
  score: number;
}

const leaderboard = new Leaderboard<Player>(
  5,
  p => p.id,
  (a, b) => a.score - b.score
);

// 添加玩家
const players: Player[] = [
  { id: 'p1', name: 'Alice', score: 100 },
  { id: 'p2', name: 'Bob', score: 85 },
  { id: 'p3', name: 'Charlie', score: 120 },
  { id: 'p4', name: 'David', score: 95 },
  { id: 'p5', name: 'Eve', score: 110 },
  { id: 'p6', name: 'Frank', score: 75 },
];

players.forEach(p => leaderboard.upsert(p));

console.log('初始排行榜:');
leaderboard.getLeaderboard().forEach((p, i) =>
  console.log(`  ${i + 1}. ${p.name}: ${p.score}`)
);

// 更新分数
leaderboard.upsert({ id: 'p2', name: 'Bob', score: 150 });
console.log('\nBob 分数更新到 150:');
leaderboard.getLeaderboard().forEach((p, i) =>
  console.log(`  ${i + 1}. ${p.name}: ${p.score}`)
);

console.log(`\nAlice 的排名: ${leaderboard.getRank('p1')}`);

// ============================================================================
// 示例 8：偏移量分页
// ============================================================================

console.log('\n=== 示例 8：偏移量分页 ===\n');

const sortedResults = topKBySort(results, 100, cmp);

const page1 = paginateByOffset(sortedResults, 1, 10);
console.log(`第 1 页 (共 ${Math.ceil(page1.total / 10)} 页):`);
printResults(page1.items, ['id', 'relevance']);

const page2 = paginateByOffset(sortedResults, 2, 10);
console.log(`\n第 2 页:`);
printResults(page2.items.slice(0, 5), ['id', 'relevance']);
console.log('  ...');

// ============================================================================
// 示例 9：游标分页
// ============================================================================

console.log('\n=== 示例 9：游标分页 ===\n');

// 第一页
const cursorPage1 = paginateWithCursor(sortedResults, null, 5, r => r.id);
console.log('游标分页 - 第 1 页:');
printResults(cursorPage1.items, ['id', 'relevance']);
console.log(`下一页游标: ${cursorPage1.nextCursor}`);

// 第二页
const cursorPage2 = paginateWithCursor(sortedResults, cursorPage1.nextCursor, 5, r => r.id);
console.log('\n游标分页 - 第 2 页:');
printResults(cursorPage2.items, ['id', 'relevance']);

console.log('\n游标分页优势: 即使数据变化，也不会漏掉或重复数据');

// ============================================================================
// 示例 10：分页管理器
// ============================================================================

console.log('\n=== 示例 10：分页管理器 ===\n');

const manager = new PaginationManager<SearchResult>(10);
manager.setData(sortedResults);

console.log('使用分页管理器:');
let currentPage = manager.getCurrentPage();
console.log(`当前页: ${currentPage.page}, 共 ${Math.ceil(currentPage.total / 10)} 页`);

manager.nextPage();
currentPage = manager.getCurrentPage();
console.log(`下一页: ${currentPage.page}`);

manager.goToPage(5);
currentPage = manager.getCurrentPage();
console.log(`跳转到第 5 页: ${currentPage.page}`);

console.log('\n✅ 所有示例运行完成');

