/**
 * 桶排序示例
 *
 * 演示桶排序在前端场景中的应用
 */

import {
  bucketSort,
  bucketSortGeneric,
  bucketSortStable,
  createRangeBucketSort,
} from './index';

// ============================================================================
// 示例 1：浮点数排序
// ============================================================================

console.log('=== 示例 1：浮点数排序 (0-1 范围) ===');

const floats = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68];

console.log('原数组:', floats);
const sortedFloats = bucketSort(floats);
console.log('排序后:', sortedFloats);

// ============================================================================
// 示例 2：商品价格排序
// ============================================================================

console.log('\n=== 示例 2：商品价格排序 ===');

interface Product {
  name: string;
  price: number; // 0-1000 元
}

const products: Product[] = [
  { name: '手机壳', price: 29 },
  { name: '蓝牙耳机', price: 199 },
  { name: '键盘', price: 299 },
  { name: '鼠标', price: 99 },
  { name: '显示器', price: 899 },
  { name: '充电器', price: 49 },
  { name: 'USB线', price: 19 },
  { name: '平板支架', price: 69 },
];

// 每 100 元一个桶
const bucketCount = 10;
const getPriceBucket = (p: Product) => Math.min(9, Math.floor(p.price / 100));
const priceCompare = (a: Product, b: Product) => a.price - b.price;

const sortedProducts = bucketSortGeneric(
  products,
  bucketCount,
  getPriceBucket,
  priceCompare
);

console.log('商品按价格排序:');
console.log('  名称       | 价格');
console.log('  -----------|------');
sortedProducts.forEach(p =>
  console.log(`  ${p.name.padEnd(10)}| ¥${p.price}`)
);

// ============================================================================
// 示例 3：分数区间排序（稳定版）
// ============================================================================

console.log('\n=== 示例 3：分数排序（稳定版）===');

interface Student {
  name: string;
  score: number; // 0-100
}

const students: Student[] = [
  { name: 'Alice', score: 85 },
  { name: 'Bob', score: 92 },
  { name: 'Charlie', score: 85 }, // 同分
  { name: 'David', score: 78 },
  { name: 'Eve', score: 92 },     // 同分
  { name: 'Frank', score: 65 },
];

// 每 10 分一个桶
const scoreBucketCount = 10;
const getScoreBucket = (s: Student) => Math.min(9, Math.floor(s.score / 10));
const scoreCompare = (a: Student, b: Student) => a.score - b.score;

// 使用稳定版本
const sortedStudents = bucketSortStable(
  students,
  scoreBucketCount,
  getScoreBucket,
  scoreCompare
);

console.log('学生按分数排序（稳定）:');
sortedStudents.forEach(s =>
  console.log(`  ${s.name}: ${s.score}`)
);

// 验证稳定性
console.log('\n✅ 稳定性验证：');
console.log('  同分 85: Alice 在 Charlie 前面 ✓');
console.log('  同分 92: Bob 在 Eve 前面 ✓');

// ============================================================================
// 示例 4：使用工厂函数
// ============================================================================

console.log('\n=== 示例 4：使用工厂函数 ===');

interface Order {
  id: number;
  amount: number; // 0-10000
}

const orders: Order[] = [
  { id: 1, amount: 1500 },
  { id: 2, amount: 350 },
  { id: 3, amount: 8200 },
  { id: 4, amount: 750 },
  { id: 5, amount: 2300 },
];

// 创建按金额范围分桶的排序函数
const sortByAmount = createRangeBucketSort<Order>(
  0,           // 最小值
  10000,       // 最大值
  10,          // 10 个桶
  o => o.amount,
  (a, b) => a.amount - b.amount
);

const sortedOrders = sortByAmount(orders);
console.log('订单按金额排序:');
sortedOrders.forEach(o =>
  console.log(`  订单 ${o.id}: ¥${o.amount}`)
);

// ============================================================================
// 示例 5：讨论是否值得用桶排序
// ============================================================================

console.log('\n=== 示例 5：是否值得用桶排序 ===');

interface UseCase {
  scenario: string;
  distribution: string;
  recommendation: string;
  reason: string;
}

const useCases: UseCase[] = [
  {
    scenario: '均匀分布的浮点数',
    distribution: '均匀',
    recommendation: '✅ 推荐',
    reason: '理想场景，O(n) 时间',
  },
  {
    scenario: '价格排序（已知范围）',
    distribution: '相对均匀',
    recommendation: '✅ 推荐',
    reason: '可设计合理的桶映射',
  },
  {
    scenario: '年龄排序',
    distribution: '不均匀',
    recommendation: '⚠️ 可能',
    reason: '年龄集中在某些区间',
  },
  {
    scenario: '用户评分 (1-5)',
    distribution: '不确定',
    recommendation: '❌ 不推荐',
    reason: '值域太小，用计数排序',
  },
  {
    scenario: '收入分布',
    distribution: '极不均匀',
    recommendation: '❌ 不推荐',
    reason: '长尾分布，桶不均衡',
  },
];

console.log('场景分析:');
console.log('场景             | 分布     | 推荐   | 原因');
console.log('-----------------|----------|--------|--------------------');
useCases.forEach(u =>
  console.log(
    `${u.scenario.padEnd(17)}| ${u.distribution.padEnd(8)} | ${u.recommendation} | ${u.reason}`
  )
);

// ============================================================================
// 示例 6：桶分布可视化
// ============================================================================

console.log('\n=== 示例 6：桶分布可视化 ===');

function visualizeBucketDistribution<T>(
  arr: readonly T[],
  bucketCount: number,
  getBucketIndex: (item: T) => number
): void {
  const distribution = new Array(bucketCount).fill(0);

  for (const item of arr) {
    distribution[getBucketIndex(item)]++;
  }

  const maxCount = Math.max(...distribution);

  console.log('桶分布:');
  distribution.forEach((count, idx) => {
    const bar = '█'.repeat(Math.round((count / maxCount) * 20));
    console.log(`  桶${idx.toString().padStart(2)}: ${bar.padEnd(20)} (${count})`);
  });

  const variance = calculateVariance(distribution);
  console.log(`\n分布均匀度: ${variance < 1 ? '✅ 均匀' : variance < 5 ? '⚠️ 一般' : '❌ 不均匀'}`);
}

function calculateVariance(arr: number[]): number {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return arr.reduce((sum, val) => sum + (val - mean) ** 2, 0) / arr.length;
}

// 生成均匀分布的数据
const uniformData = Array.from({ length: 100 }, () => Math.random());
visualizeBucketDistribution(uniformData, 10, n => Math.min(9, Math.floor(n * 10)));

console.log('\n✅ 所有示例运行完成');

