/**
 * TimSort - 使用示例
 *
 * 重点展示：
 * 1. 近乎有序数据的优势
 * 2. 表格多列稳定排序
 */

import { sort, sortWithStats, meta } from './index';
import { numberAsc, byField, reverse, compose } from '../../公共库/src/比较器';
import { generateNumbers } from '../../公共库/src/数据生成器';

console.log(`\n===== ${meta.name} Demo =====\n`);

// ============================================================================
// 1. 近乎有序数据的优势
// ============================================================================

console.log('--- 1. 近乎有序数据的优势 ---');

const nearlySorted = generateNumbers(10000, 'nearlySorted', { swapPercent: 1 });
const random = generateNumbers(10000, 'random');

console.log('数据规模: 10000 个元素');

console.time('  近乎有序数据');
sort(nearlySorted, numberAsc);
console.timeEnd('  近乎有序数据');

console.time('  完全随机数据');
sort(random, numberAsc);
console.timeEnd('  完全随机数据');

console.log('\n⭐ TimSort 在近乎有序数据上明显更快！');

// ============================================================================
// 2. Run 检测统计
// ============================================================================

console.log('\n--- 2. Run 检测统计 ---');

const ordered = Array.from({ length: 1000 }, (_, i) => i);
const orderedStats = sortWithStats(ordered, numberAsc);
console.log(`完全有序 (1000): runs=${orderedStats.runs}, merges=${orderedStats.merges}`);

const nearlyStats = sortWithStats(
  generateNumbers(1000, 'nearlySorted', { swapPercent: 5 }),
  numberAsc
);
console.log(`近乎有序 (1000): runs=${nearlyStats.runs}, merges=${nearlyStats.merges}`);

const randomStats = sortWithStats(generateNumbers(1000, 'random'), numberAsc);
console.log(`完全随机 (1000): runs=${randomStats.runs}, merges=${randomStats.merges}`);

console.log('\n⭐ 有序数据的 run 数量少，合并次数也少！');

// ============================================================================
// 3. 表格多列稳定排序 ⭐
// ============================================================================

console.log('\n--- 3. 表格多列稳定排序 ⭐ ---');

interface Employee {
  id: number;
  name: string;
  department: string;
  salary: number;
  joinDate: string;
}

const employees: Employee[] = [
  { id: 1, name: 'Alice', department: 'Dev', salary: 80000, joinDate: '2021-01' },
  { id: 2, name: 'Bob', department: 'HR', salary: 60000, joinDate: '2020-06' },
  { id: 3, name: 'Charlie', department: 'Dev', salary: 75000, joinDate: '2021-01' },
  { id: 4, name: 'David', department: 'Dev', salary: 80000, joinDate: '2020-06' },
  { id: 5, name: 'Eve', department: 'HR', salary: 65000, joinDate: '2021-01' },
  { id: 6, name: 'Frank', department: 'Dev', salary: 70000, joinDate: '2020-06' },
  { id: 7, name: 'Grace', department: 'HR', salary: 60000, joinDate: '2021-01' },
  { id: 8, name: 'Henry', department: 'Dev', salary: 85000, joinDate: '2019-03' },
];

console.log('原始数据:');
employees.forEach((e) =>
  console.log(`  ${e.name.padEnd(8)} | ${e.department} | $${e.salary} | ${e.joinDate}`)
);

// 第一步：按入职时间排序
const byJoinDate = (a: Employee, b: Employee) => a.joinDate.localeCompare(b.joinDate);
let sorted = sort(employees, byJoinDate);

console.log('\n第一步 - 按入职时间排序:');
sorted.forEach((e) =>
  console.log(`  ${e.name.padEnd(8)} | ${e.department} | $${e.salary} | ${e.joinDate}`)
);

// 第二步：按部门排序（稳定性保证同部门的入职顺序）
const byDepartment = (a: Employee, b: Employee) => a.department.localeCompare(b.department);
sorted = sort(sorted, byDepartment);

console.log('\n第二步 - 按部门排序（同部门保持入职顺序）:');
sorted.forEach((e) =>
  console.log(`  ${e.name.padEnd(8)} | ${e.department} | $${e.salary} | ${e.joinDate}`)
);

// 第三步：按薪资降序排序（稳定性保证同薪资的部门和入职顺序）
const bySalaryDesc = (a: Employee, b: Employee) => b.salary - a.salary;
sorted = sort(sorted, bySalaryDesc);

console.log('\n第三步 - 按薪资降序（同薪资保持之前的顺序）:');
sorted.forEach((e) =>
  console.log(`  ${e.name.padEnd(8)} | ${e.department} | $${e.salary} | ${e.joinDate}`)
);

console.log('\n⭐ 稳定排序让多列排序变得简单可靠！');

// ============================================================================
// 4. 时间序列数据
// ============================================================================

console.log('\n--- 4. 时间序列数据（通常部分有序）---');

interface LogEntry {
  timestamp: number;
  message: string;
}

// 模拟日志数据：大部分有序，偶尔有乱序（如并发写入）
const logs: LogEntry[] = Array.from({ length: 100 }, (_, i) => ({
  timestamp: i * 100 + (Math.random() > 0.9 ? -50 : 0), // 10% 概率乱序
  message: `Log ${i}`,
}));

console.log('日志数据（部分乱序）:');
console.log(`  前5条时间戳: ${logs.slice(0, 5).map((l) => l.timestamp).join(', ')}`);

const sortedLogs = sort(logs, (a, b) => a.timestamp - b.timestamp);
console.log(`  排序后前5条: ${sortedLogs.slice(0, 5).map((l) => l.timestamp).join(', ')}`);

// ============================================================================
// 5. 搜索结果排序
// ============================================================================

console.log('\n--- 5. 搜索结果排序 ---');

interface SearchResult {
  id: string;
  title: string;
  relevance: number;
  publishTime: number;
}

const searchResults: SearchResult[] = [
  { id: 'a', title: 'Node.js Tutorial', relevance: 0.85, publishTime: 1700000100 },
  { id: 'b', title: 'React Hooks Guide', relevance: 0.92, publishTime: 1700000200 },
  { id: 'c', title: 'Vue 3 Basics', relevance: 0.85, publishTime: 1700000300 },
  { id: 'd', title: 'TypeScript Deep Dive', relevance: 0.92, publishTime: 1700000050 },
  { id: 'e', title: 'Node.js Performance', relevance: 0.85, publishTime: 1700000150 },
];

console.log('原始搜索结果:');
searchResults.forEach((r) => console.log(`  [${r.relevance}] ${r.title}`));

// 按相关度降序，同相关度按时间降序
const byRelevanceDesc = (a: SearchResult, b: SearchResult) => b.relevance - a.relevance;
const byTimeDesc = (a: SearchResult, b: SearchResult) => b.publishTime - a.publishTime;

// 先按时间排序，再按相关度排序（利用稳定性）
let sortedResults = sort(searchResults, byTimeDesc);
sortedResults = sort(sortedResults, byRelevanceDesc);

console.log('\n按相关度降序，同分按时间降序:');
sortedResults.forEach((r) => console.log(`  [${r.relevance}] ${r.title}`));

// ============================================================================
// 6. 不同数据分布的表现
// ============================================================================

console.log('\n--- 6. 不同数据分布的表现 ---');

const distributions = [
  { name: 'sorted', label: '完全有序' },
  { name: 'nearlySorted', label: '近乎有序' },
  { name: 'random', label: '完全随机' },
  { name: 'reversed', label: '完全逆序' },
] as const;

for (const { name, label } of distributions) {
  const data = generateNumbers(10000, name as any);

  console.time(`  ${label}`);
  sort(data, numberAsc);
  console.timeEnd(`  ${label}`);
}

console.log('\n⭐ 完全有序和近乎有序时最快，逆序也不慢（反转 run）');

// ============================================================================
// 7. 元信息展示
// ============================================================================

console.log('\n--- 7. 算法信息 ---');
console.log(`名称: ${meta.name}`);
console.log(`稳定性: ${meta.stable ? '✅ 稳定' : '❌ 不稳定'}`);
console.log(`时间复杂度（最好）: ${meta.timeComplexity.best}`);
console.log(`使用者: ${meta.使用者.join(', ')}`);

console.log('\n===== Demo 完成 =====');

