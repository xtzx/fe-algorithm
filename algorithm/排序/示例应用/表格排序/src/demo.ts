/**
 * 表格排序综合示例
 */

import { SAMPLE_DATA, printTable, type TableRow, type SortColumn } from './数据模型';
import {
  sortByMultipleColumns,
  handleSingleColumnSort,
  handleMultiColumnSort,
} from './多列稳定排序';
import {
  createTableComparator,
  TableComparatorBuilder,
  sortByComparator,
} from './比较器组合器';

// ============================================================================
// 示例 1：基础多列排序
// ============================================================================

console.log('=== 示例 1：基础多列排序 ===\n');

console.log('原始数据:');
printTable(SAMPLE_DATA);

// 方式 A：多列稳定排序
console.log('\n方式 A：多列稳定排序（先按部门，再按分数降序）');
const sortedA = sortByMultipleColumns(SAMPLE_DATA, [
  { field: 'department', order: 'asc', type: 'string' },
  { field: 'score', order: 'desc', type: 'number' },
]);
printTable(sortedA, ['name', 'department', 'score']);

// 方式 B：比较器组合
console.log('\n方式 B：比较器组合（相同排序条件）');
const sortedB = sortByComparator(SAMPLE_DATA, [
  { field: 'department', order: 'asc', type: 'string' },
  { field: 'score', order: 'desc', type: 'number' },
]);
printTable(sortedB, ['name', 'department', 'score']);

// ============================================================================
// 示例 2：验证稳定性
// ============================================================================

console.log('\n=== 示例 2：验证稳定性 ===\n');

// 先按分数排序
const byScore = sortByMultipleColumns(SAMPLE_DATA, [
  { field: 'score', order: 'desc', type: 'number' },
]);
console.log('第一步：按分数降序');
printTable(byScore, ['name', 'score', 'department']);

// 再按部门排序（稳定排序保持分数顺序）
const byDeptThenScore = sortByMultipleColumns(byScore, [
  { field: 'department', order: 'asc', type: 'string' },
]);
console.log('\n第二步：按部门升序（同部门保持分数降序）');
printTable(byDeptThenScore, ['name', 'department', 'score']);

console.log('\n✅ 验证：同部门的员工仍按分数降序排列');

// ============================================================================
// 示例 3：动态列排序（模拟点击）
// ============================================================================

console.log('\n=== 示例 3：模拟点击列排序 ===\n');

let state = {
  data: SAMPLE_DATA,
  columns: [] as SortColumn<TableRow>[],
};

console.log('初始状态：无排序');

// 点击 score 列
state = handleMultiColumnSort(state.data, state.columns, 'score', 'number', false);
console.log('\n点击 score 列（升序）:');
console.log('当前排序:', state.columns.map(c => `${String(c.field)} ${c.order}`).join(', '));
printTable(state.data.slice(0, 5), ['name', 'score']);

// 再次点击 score 列（降序）
state = handleMultiColumnSort(state.data, state.columns, 'score', 'number', false);
console.log('\n再次点击 score 列（降序）:');
console.log('当前排序:', state.columns.map(c => `${String(c.field)} ${c.order}`).join(', '));
printTable(state.data.slice(0, 5), ['name', 'score']);

// Shift+点击 department 列（添加次级排序）
state = handleMultiColumnSort(state.data, state.columns, 'department', 'string', true);
console.log('\nShift+点击 department 列（添加次级排序）:');
console.log('当前排序:', state.columns.map(c => `${String(c.field)} ${c.order}`).join(', '));
printTable(state.data, ['name', 'department', 'score']);

// ============================================================================
// 示例 4：使用构建器
// ============================================================================

console.log('\n=== 示例 4：使用 TableComparatorBuilder ===\n');

const builder = new TableComparatorBuilder<TableRow>()
  .addColumn('department', 'asc', 'string')
  .addColumn('salary', 'desc', 'number');

console.log('构建器配置:', builder.getColumns());

const sortedByBuilder = builder.sort(SAMPLE_DATA);
console.log('\n排序结果（部门升序，工资降序）:');
printTable(sortedByBuilder, ['name', 'department', 'salary']);

// 动态修改
builder.toggleOrder('department');
console.log('\n切换部门为降序:');
printTable(builder.sort(SAMPLE_DATA).slice(0, 5), ['name', 'department', 'salary']);

// ============================================================================
// 示例 5：日期排序
// ============================================================================

console.log('\n=== 示例 5：日期排序 ===\n');

const byJoinDate = sortByComparator(SAMPLE_DATA, [
  { field: 'joinDate', order: 'asc', type: 'date' },
]);

console.log('按入职日期升序:');
printTable(byJoinDate, ['name', 'joinDate', 'department']);

// ============================================================================
// 示例 6：性能对比
// ============================================================================

console.log('\n=== 示例 6：性能对比 ===\n');

import { generateTableData } from './数据模型';

const largeData = generateTableData(10000);

// 方式 A：多列稳定排序
const startA = performance.now();
sortByMultipleColumns(largeData, [
  { field: 'department', order: 'asc', type: 'string' },
  { field: 'score', order: 'desc', type: 'number' },
  { field: 'salary', order: 'asc', type: 'number' },
]);
const durationA = performance.now() - startA;

// 方式 B：比较器组合
const startB = performance.now();
sortByComparator(largeData, [
  { field: 'department', order: 'asc', type: 'string' },
  { field: 'score', order: 'desc', type: 'number' },
  { field: 'salary', order: 'asc', type: 'number' },
]);
const durationB = performance.now() - startB;

console.log(`数据量: ${largeData.length} 条，排序 3 列`);
console.log(`方式 A（多列稳定排序）: ${durationA.toFixed(2)} ms`);
console.log(`方式 B（比较器组合）: ${durationB.toFixed(2)} ms`);
console.log(`性能差异: 方式 B 快 ${((durationA - durationB) / durationA * 100).toFixed(1)}%`);

// ============================================================================
// 示例 7：处理边界情况
// ============================================================================

console.log('\n=== 示例 7：边界情况 ===\n');

// 空数组
const emptyResult = sortByMultipleColumns([], [{ field: 'score', order: 'asc' }]);
console.log('空数组排序:', emptyResult);

// 无排序列
const noSortResult = sortByMultipleColumns(SAMPLE_DATA.slice(0, 3), []);
console.log('无排序列:', noSortResult.map(r => r.name));

// 单元素
const singleResult = sortByMultipleColumns([SAMPLE_DATA[0]], [{ field: 'score', order: 'asc' }]);
console.log('单元素:', singleResult.map(r => r.name));

console.log('\n✅ 所有示例运行完成');

