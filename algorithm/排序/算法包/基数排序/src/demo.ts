/**
 * 基数排序示例
 *
 * 演示基数排序在前端场景中的应用
 */

import {
  radixSort,
  radixSortBy,
  radixSortWithNegative,
  radixSortByWithNegative,
} from './index';

// ============================================================================
// 示例 1：基础整数排序
// ============================================================================

console.log('=== 示例 1：基础整数排序 ===');

const numbers = [170, 45, 75, 90, 802, 24, 2, 66];
console.log('原数组:', numbers);

const sorted = radixSort(numbers);
console.log('排序后:', sorted);

// 展示排序过程
console.log('\n排序过程（LSD）:');
console.log('  第1轮（个位）: [170, 90, 802, 2, 24, 45, 75, 66]');
console.log('  第2轮（十位）: [802, 2, 24, 45, 66, 170, 75, 90]');
console.log('  第3轮（百位）: [2, 24, 45, 66, 75, 90, 170, 802]');

// ============================================================================
// 示例 2：手机号排序
// ============================================================================

console.log('\n=== 示例 2：手机号排序 ===');

interface Contact {
  name: string;
  phone: number; // 11位手机号
}

const contacts: Contact[] = [
  { name: 'Alice', phone: 13812345678 },
  { name: 'Bob', phone: 15987654321 },
  { name: 'Charlie', phone: 13698765432 },
  { name: 'David', phone: 18712341234 },
  { name: 'Eve', phone: 13512345678 },
];

// 按手机号排序
const sortedContacts = radixSortBy(contacts, c => c.phone);

console.log('按手机号排序:');
sortedContacts.forEach(c =>
  console.log(`  ${c.name}: ${c.phone}`)
);

// ============================================================================
// 示例 3：订单号排序
// ============================================================================

console.log('\n=== 示例 3：订单号排序 ===');

interface Order {
  id: number;
  product: string;
  timestamp: number;
}

const orders: Order[] = [
  { id: 20230001, product: '商品A', timestamp: 1672531200000 },
  { id: 20220050, product: '商品B', timestamp: 1640995200000 },
  { id: 20230100, product: '商品C', timestamp: 1680307200000 },
  { id: 20210005, product: '商品D', timestamp: 1609459200000 },
];

// 按订单号排序（稳定）
const sortedOrders = radixSortBy(orders, o => o.id);

console.log('按订单号排序:');
console.log('  订单号   | 商品');
console.log('  ---------|------');
sortedOrders.forEach(o =>
  console.log(`  ${o.id} | ${o.product}`)
);

// ============================================================================
// 示例 4：负数排序
// ============================================================================

console.log('\n=== 示例 4：负数排序 ===');

const withNegative = [-5, 3, -2, 8, -1, 0, 5, -3];
console.log('原数组:', withNegative);

const sortedWithNeg = radixSortWithNegative(withNegative);
console.log('排序后:', sortedWithNeg);

// ============================================================================
// 示例 5：温度数据排序
// ============================================================================

console.log('\n=== 示例 5：温度数据排序（含负值）===');

interface TemperatureRecord {
  city: string;
  temp: number; // 摄氏度，可能为负
}

const temperatures: TemperatureRecord[] = [
  { city: '北京', temp: -5 },
  { city: '上海', temp: 8 },
  { city: '哈尔滨', temp: -20 },
  { city: '广州', temp: 15 },
  { city: '三亚', temp: 25 },
  { city: '乌鲁木齐', temp: -10 },
];

const sortedTemps = radixSortByWithNegative(temperatures, t => t.temp);

console.log('按温度排序:');
sortedTemps.forEach(t =>
  console.log(`  ${t.city}: ${t.temp}°C`)
);

// ============================================================================
// 示例 6：表格数据排序
// ============================================================================

console.log('\n=== 示例 6：表格数据排序 ===');

interface TableRow {
  id: number;
  name: string;
  score: number;
  rank: number;
}

const tableData: TableRow[] = [
  { id: 1001, name: 'Alice', score: 95, rank: 1 },
  { id: 1005, name: 'Bob', score: 88, rank: 3 },
  { id: 1003, name: 'Charlie', score: 92, rank: 2 },
  { id: 1002, name: 'David', score: 85, rank: 4 },
  { id: 1004, name: 'Eve', score: 95, rank: 1 }, // 同分
];

// 按 ID 排序
const byId = radixSortBy(tableData, r => r.id);
console.log('按 ID 排序:');
console.log('  ID   | Name    | Score');
console.log('  -----|---------|------');
byId.forEach(r =>
  console.log(`  ${r.id} | ${r.name.padEnd(7)} | ${r.score}`)
);

// 验证稳定性：按 score 排序后，相同 score 的保持原 ID 顺序
const byScore = radixSortBy(tableData, r => r.score);
console.log('\n按分数排序（验证稳定性）:');
byScore.forEach(r =>
  console.log(`  ${r.name}: ${r.score} (ID: ${r.id})`)
);
console.log('✅ 同分的 Alice(1001) 在 Eve(1004) 前面');

// ============================================================================
// 示例 7：讨论是否值得用基数排序
// ============================================================================

console.log('\n=== 示例 7：是否值得用基数排序 ===');

interface UseCase {
  scenario: string;
  digits: string;
  recommendation: string;
  reason: string;
}

const useCases: UseCase[] = [
  {
    scenario: '手机号 (11位)',
    digits: '11',
    recommendation: '✅ 推荐',
    reason: '固定位数，O(11n) < O(n log n)',
  },
  {
    scenario: '订单号 (8位)',
    digits: '8',
    recommendation: '✅ 推荐',
    reason: '固定位数，稳定高效',
  },
  {
    scenario: '用户 ID (不定长)',
    digits: '变化大',
    recommendation: '⚠️ 考虑',
    reason: '位数变化可能影响效率',
  },
  {
    scenario: '金额 (0-10亿)',
    digits: '最多10',
    recommendation: '✅ 推荐',
    reason: '位数可控，稳定',
  },
  {
    scenario: '浮点数',
    digits: 'N/A',
    recommendation: '❌ 不适用',
    reason: '需要特殊处理位表示',
  },
  {
    scenario: '小范围整数 (0-100)',
    digits: '3',
    recommendation: '⚠️ 用计数排序',
    reason: '计数排序更简单直接',
  },
];

console.log('场景分析:');
console.log('场景             | 位数   | 推荐   | 原因');
console.log('-----------------|--------|--------|--------------------');
useCases.forEach(u =>
  console.log(
    `${u.scenario.padEnd(17)}| ${u.digits.padEnd(6)} | ${u.recommendation} | ${u.reason}`
  )
);

// ============================================================================
// 示例 8：基数选择的影响
// ============================================================================

console.log('\n=== 示例 8：基数选择的影响 ===');

const largeNumbers = [12345678, 87654321, 11111111, 99999999];

console.log('原数组:', largeNumbers);

// 基数 10（默认）
console.log('基数 10: 需要 8 轮');

// 基数 256（二进制优化）
console.log('基数 256: 只需要 4 轮（但每轮桶更多）');

console.log('\n基数选择建议:');
console.log('  - 基数 10: 直观，适合调试');
console.log('  - 基数 256: 位运算优化，适合性能敏感场景');

console.log('\n✅ 所有示例运行完成');

