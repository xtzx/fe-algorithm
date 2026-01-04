/**
 * 归并排序 - 测试用例
 */

import { sort, sortInPlace, sortIterative, sortOptimized, merge, meta } from '../src/index';
import { verifySort } from '../../公共库/src/正确性校验';
import { numberAsc, byField, compose } from '../../公共库/src/比较器';
import {
  generateNumbers,
  generateTableRows,
  generateBoundaryTestCases,
} from '../../公共库/src/数据生成器';

console.log(`\n===== ${meta.name} 测试 =====\n`);

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void): void {
  try {
    fn();
    console.log(`✅ ${name}`);
    passed++;
  } catch (error) {
    console.log(`❌ ${name}`);
    console.error(`   ${(error as Error).message}`);
    failed++;
  }
}

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

// ============================================================================
// 边界测试
// ============================================================================

console.log('--- 边界测试 ---');

const boundaryTests = generateBoundaryTestCases();

for (const { name, data } of boundaryTests) {
  test(`边界: ${name}`, () => {
    const original = [...data];
    const sorted = sort(data, numberAsc);
    const result = verifySort(original, sorted, numberAsc, true); // 检查稳定性
    assert(result.passed, result.error || '未知错误');
  });
}

// ============================================================================
// 各种数据分布测试
// ============================================================================

console.log('\n--- 数据分布测试 ---');

const distributions = ['random', 'sorted', 'reversed', 'nearlySorted', 'fewUnique'] as const;

for (const dist of distributions) {
  test(`分布: ${dist} (n=1000)`, () => {
    const data = generateNumbers(1000, dist);
    const original = [...data];
    const sorted = sort(data, numberAsc);
    const result = verifySort(original, sorted, numberAsc, true);
    assert(result.passed, result.error || '未知错误');
  });
}

// ============================================================================
// 原地排序测试
// ============================================================================

console.log('\n--- 原地排序测试 ---');

test('sortInPlace 修改原数组', () => {
  const arr = [5, 3, 1, 4, 2];
  const original = [...arr];
  const result = sortInPlace(arr, numberAsc);
  assert(result === arr, '应返回同一引用');
  const verification = verifySort(original, arr, numberAsc, true);
  assert(verification.passed, verification.error || '未知错误');
});

// ============================================================================
// 迭代版测试
// ============================================================================

console.log('\n--- 迭代版测试 ---');

test('sortIterative 正确性', () => {
  const data = generateNumbers(500, 'random');
  const original = [...data];
  const sorted = sortIterative(data, numberAsc);
  const result = verifySort(original, sorted, numberAsc, true);
  assert(result.passed, result.error || '未知错误');
});

test('sortIterative 与递归版结果一致', () => {
  const data = generateNumbers(200, 'random');
  const sorted1 = sort(data, numberAsc);
  const sorted2 = sortIterative(data, numberAsc);
  assert(
    JSON.stringify(sorted1) === JSON.stringify(sorted2),
    '递归版和迭代版结果应一致'
  );
});

// ============================================================================
// 优化版测试
// ============================================================================

console.log('\n--- 优化版测试 ---');

test('sortOptimized 正确性', () => {
  const data = generateNumbers(1000, 'random');
  const original = [...data];
  const sorted = sortOptimized(data, numberAsc);
  const result = verifySort(original, sorted, numberAsc, true);
  assert(result.passed, result.error || '未知错误');
});

// ============================================================================
// 稳定性测试
// ============================================================================

console.log('\n--- 稳定性测试 ---');

test('稳定性: 相等元素保持原顺序', () => {
  interface Item {
    value: number;
    originalIndex: number;
  }

  const items: Item[] = [
    { value: 3, originalIndex: 0 },
    { value: 1, originalIndex: 1 },
    { value: 3, originalIndex: 2 },
    { value: 2, originalIndex: 3 },
    { value: 3, originalIndex: 4 },
  ];

  const cmp = (a: Item, b: Item) => a.value - b.value;
  const sorted = sort(items, cmp);

  // 检查 value=3 的元素是否保持原顺序
  const threes = sorted.filter((item) => item.value === 3);
  assert(threes[0].originalIndex === 0, '第一个 3 应是 originalIndex=0');
  assert(threes[1].originalIndex === 2, '第二个 3 应是 originalIndex=2');
  assert(threes[2].originalIndex === 4, '第三个 3 应是 originalIndex=4');
});

// ============================================================================
// 对象排序测试
// ============================================================================

console.log('\n--- 对象排序测试 ---');

test('表格数据排序', () => {
  const rows = generateTableRows(100);
  const original = [...rows];

  const cmp = compose(
    byField<typeof rows[0], 'grade'>('grade', (a, b) => a.localeCompare(b)),
    byField<typeof rows[0], 'score'>('score', (a, b) => b - a) // 分数降序
  );

  const sorted = sort(rows, cmp);
  const result = verifySort(original, sorted, cmp, true);
  assert(result.passed, result.error || '未知错误');
});

// ============================================================================
// merge 函数测试
// ============================================================================

console.log('\n--- merge 函数测试 ---');

test('merge 两个有序数组', () => {
  const left = [1, 3, 5, 7];
  const right = [2, 4, 6, 8];
  const merged = merge(left, right, numberAsc);
  assert(
    JSON.stringify(merged) === JSON.stringify([1, 2, 3, 4, 5, 6, 7, 8]),
    '合并结果错误'
  );
});

test('merge 空数组', () => {
  assert(
    JSON.stringify(merge([], [1, 2, 3], numberAsc)) === JSON.stringify([1, 2, 3]),
    '左空错误'
  );
  assert(
    JSON.stringify(merge([1, 2, 3], [], numberAsc)) === JSON.stringify([1, 2, 3]),
    '右空错误'
  );
  assert(
    JSON.stringify(merge([], [], numberAsc)) === JSON.stringify([]),
    '双空错误'
  );
});

// ============================================================================
// 元信息测试
// ============================================================================

console.log('\n--- 元信息测试 ---');

test('meta 信息正确', () => {
  assert(meta.stable === true, '应为稳定排序');
  assert(meta.inPlace === false, '不是原地排序');
  assert(meta.timeComplexity.worst === 'O(n log n)', '最坏时间复杂度');
  assert(meta.spaceComplexity === 'O(n)', '空间复杂度');
});

// ============================================================================
// 总结
// ============================================================================

console.log(`\n===== 测试完成 =====`);
console.log(`✅ 通过: ${passed}`);
console.log(`❌ 失败: ${failed}`);

if (failed > 0) {
  process.exit(1);
}

