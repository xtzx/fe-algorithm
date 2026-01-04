/**
 * 快速排序 - 测试用例
 */

import {
  sort,
  sortInPlace,
  sortRobust,
  sortWithStats,
  partition,
  partitionRandom,
  meta,
} from '../src/index';
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
    // 不检查稳定性，因为快排不稳定
    const result = verifySort(original, sorted, numberAsc, false);
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
    const sorted = sortRobust(data, numberAsc); // 用健壮版避免最坏情况
    const result = verifySort(original, sorted, numberAsc, false);
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
  const verification = verifySort(original, arr, numberAsc, false);
  assert(verification.passed, verification.error || '未知错误');
});

// ============================================================================
// 健壮版测试
// ============================================================================

console.log('\n--- 健壮版测试 ---');

test('sortRobust 处理有序数据', () => {
  const data = generateNumbers(2000, 'sorted');
  const original = [...data];
  const sorted = sortRobust(data, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

test('sortRobust 处理逆序数据', () => {
  const data = generateNumbers(2000, 'reversed');
  const original = [...data];
  const sorted = sortRobust(data, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

test('sortRobust 处理大量重复', () => {
  const data = generateNumbers(2000, 'fewUnique');
  const original = [...data];
  const sorted = sortRobust(data, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

// ============================================================================
// partition 测试
// ============================================================================

console.log('\n--- partition 测试 ---');

test('partition 正确分区', () => {
  const arr = [3, 8, 2, 5, 1, 4, 7, 6];
  const pivotIndex = partition(arr, 0, arr.length - 1, numberAsc);

  // pivot 左边都小于 pivot，右边都大于等于 pivot
  const pivot = arr[pivotIndex];
  for (let i = 0; i < pivotIndex; i++) {
    assert(arr[i] < pivot, `左边元素 ${arr[i]} 应小于 pivot ${pivot}`);
  }
  for (let i = pivotIndex + 1; i < arr.length; i++) {
    assert(arr[i] >= pivot, `右边元素 ${arr[i]} 应大于等于 pivot ${pivot}`);
  }
});

test('partitionRandom 正确分区', () => {
  const arr = [3, 8, 2, 5, 1, 4, 7, 6];
  const pivotIndex = partitionRandom(arr, 0, arr.length - 1, numberAsc);

  const pivot = arr[pivotIndex];
  for (let i = 0; i < pivotIndex; i++) {
    assert(arr[i] < pivot, `左边元素应小于 pivot`);
  }
  for (let i = pivotIndex + 1; i < arr.length; i++) {
    assert(arr[i] >= pivot, `右边元素应大于等于 pivot`);
  }
});

// ============================================================================
// 对象排序测试
// ============================================================================

console.log('\n--- 对象排序测试 ---');

test('表格数据排序', () => {
  const rows = generateTableRows(100);
  const original = [...rows];

  const cmp = compose(
    byField<(typeof rows)[0], 'grade'>('grade', (a, b) => a.localeCompare(b)),
    byField<(typeof rows)[0], 'score'>('score', (a, b) => b - a)
  );

  const sorted = sort(rows, cmp);
  const result = verifySort(original, sorted, cmp, false);
  assert(result.passed, result.error || '未知错误');
});

// ============================================================================
// 统计测试
// ============================================================================

console.log('\n--- 统计测试 ---');

test('sortWithStats 返回正确统计', () => {
  const data = [5, 3, 8, 4, 2, 7, 1, 6];
  const { result, comparisons, swaps, recursionDepth } = sortWithStats(
    data,
    numberAsc
  );

  assert(
    JSON.stringify(result) === JSON.stringify([1, 2, 3, 4, 5, 6, 7, 8]),
    '排序结果错误'
  );
  assert(comparisons > 0, '比较次数应大于 0');
  assert(swaps > 0, '交换次数应大于 0');
  assert(recursionDepth > 0, '递归深度应大于 0');
});

// ============================================================================
// 不稳定性测试
// ============================================================================

console.log('\n--- 不稳定性说明 ---');

test('快排不保证稳定性', () => {
  // 这个测试只是说明快排不稳定，不是断言失败
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

  // 验证排序正确
  const verification = verifySort(items, sorted, cmp, false);
  assert(verification.passed, '排序应该正确');

  // 说明：相等元素的顺序可能改变
  console.log('   注意：快排不保证相等元素的相对顺序');
});

// ============================================================================
// 元信息测试
// ============================================================================

console.log('\n--- 元信息测试 ---');

test('meta 信息正确', () => {
  assert(meta.stable === false, '应为不稳定排序');
  assert(meta.inPlace === true, '应为原地排序');
  assert(meta.timeComplexity.worst === 'O(n²)', '最坏时间复杂度');
  assert(meta.spaceComplexity === 'O(log n)', '空间复杂度');
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

