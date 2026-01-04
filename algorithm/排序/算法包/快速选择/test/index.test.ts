/**
 * 快速选择 - 测试用例
 */

import {
  quickSelect,
  quickSelectCopy,
  topKSmallest,
  topKLargest,
  topKSmallestSorted,
  topKLargestSorted,
  kthSmallest,
  kthLargest,
  median,
  percentile,
  meta,
} from '../src/index';
import { numberAsc } from '../../公共库/src/比较器';
import { generateNumbers } from '../../公共库/src/数据生成器';

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

function assertEq<T>(actual: T, expected: T, message: string): void {
  if (actual !== expected) {
    throw new Error(`${message}: 期望 ${expected}，实际 ${actual}`);
  }
}

// ============================================================================
// quickSelect 基础测试
// ============================================================================

console.log('--- quickSelect 基础测试 ---');

test('quickSelect: 找最小值', () => {
  const arr = [3, 1, 4, 1, 5, 9, 2, 6];
  const result = quickSelectCopy(arr, 0, numberAsc);
  assertEq(result, 1, '最小值应为 1');
});

test('quickSelect: 找最大值', () => {
  const arr = [3, 1, 4, 1, 5, 9, 2, 6];
  const result = quickSelectCopy(arr, arr.length - 1, numberAsc);
  assertEq(result, 9, '最大值应为 9');
});

test('quickSelect: 找中间值', () => {
  const arr = [5, 3, 1, 4, 2];
  // 排序后: [1, 2, 3, 4, 5]
  const result = quickSelectCopy(arr, 2, numberAsc);
  assertEq(result, 3, '中间值应为 3');
});

test('quickSelect: 单元素数组', () => {
  const arr = [42];
  const result = quickSelectCopy(arr, 0, numberAsc);
  assertEq(result, 42, '单元素应为 42');
});

test('quickSelect: 两元素数组', () => {
  const arr = [5, 3];
  assertEq(quickSelectCopy(arr, 0, numberAsc), 3, '第 1 小应为 3');
  assertEq(quickSelectCopy(arr, 1, numberAsc), 5, '第 2 小应为 5');
});

test('quickSelect: 含重复元素', () => {
  const arr = [3, 3, 3, 1, 1, 2, 2];
  // 排序后: [1, 1, 2, 2, 3, 3, 3]
  assertEq(quickSelectCopy(arr, 0, numberAsc), 1, '第 1 小');
  assertEq(quickSelectCopy(arr, 2, numberAsc), 2, '第 3 小');
  assertEq(quickSelectCopy(arr, 6, numberAsc), 3, '第 7 小');
});

test('quickSelect: k 越界抛出错误', () => {
  const arr = [1, 2, 3];
  let threw = false;
  try {
    quickSelectCopy(arr, 5, numberAsc);
  } catch {
    threw = true;
  }
  assert(threw, '应抛出错误');
});

// ============================================================================
// TopK 测试
// ============================================================================

console.log('\n--- TopK 测试 ---');

test('topKSmallest: 基础用例', () => {
  const arr = [5, 3, 8, 4, 2, 7, 1, 6];
  const result = topKSmallest(arr, 3, numberAsc);
  const sorted = result.sort((a, b) => a - b);
  assert(
    JSON.stringify(sorted) === JSON.stringify([1, 2, 3]),
    `最小的 3 个应为 [1, 2, 3]，实际 ${JSON.stringify(sorted)}`
  );
});

test('topKLargest: 基础用例', () => {
  const arr = [5, 3, 8, 4, 2, 7, 1, 6];
  const result = topKLargest(arr, 3, numberAsc);
  const sorted = result.sort((a, b) => b - a);
  assert(
    JSON.stringify(sorted) === JSON.stringify([8, 7, 6]),
    `最大的 3 个应为 [8, 7, 6]，实际 ${JSON.stringify(sorted)}`
  );
});

test('topKSmallestSorted: 返回有序结果', () => {
  const arr = [5, 3, 8, 4, 2];
  const result = topKSmallestSorted(arr, 3, numberAsc);
  assert(
    JSON.stringify(result) === JSON.stringify([2, 3, 4]),
    '应返回有序的 [2, 3, 4]'
  );
});

test('topKLargestSorted: 返回有序结果（降序）', () => {
  const arr = [5, 3, 8, 4, 2];
  const result = topKLargestSorted(arr, 3, numberAsc);
  assert(
    JSON.stringify(result) === JSON.stringify([8, 5, 4]),
    '应返回降序的 [8, 5, 4]'
  );
});

test('TopK: k=0 返回空数组', () => {
  const arr = [1, 2, 3];
  assert(topKSmallest(arr, 0, numberAsc).length === 0, 'k=0 应返回空数组');
});

test('TopK: k >= n 返回全部', () => {
  const arr = [1, 2, 3];
  assert(topKSmallest(arr, 5, numberAsc).length === 3, 'k>=n 应返回全部');
});

test('TopK: 不修改原数组', () => {
  const arr = [3, 1, 4, 1, 5];
  const original = JSON.stringify(arr);
  topKSmallest(arr, 2, numberAsc);
  assert(JSON.stringify(arr) === original, '原数组不应被修改');
});

// ============================================================================
// kthSmallest/kthLargest 测试（1-indexed）
// ============================================================================

console.log('\n--- kthSmallest/kthLargest 测试 ---');

test('kthSmallest: 1-indexed', () => {
  const arr = [3, 2, 1, 5, 6, 4];
  assertEq(kthSmallest(arr, 1, numberAsc), 1, '第 1 小');
  assertEq(kthSmallest(arr, 2, numberAsc), 2, '第 2 小');
  assertEq(kthSmallest(arr, 6, numberAsc), 6, '第 6 小');
});

test('kthLargest: 1-indexed', () => {
  const arr = [3, 2, 1, 5, 6, 4];
  assertEq(kthLargest(arr, 1, numberAsc), 6, '第 1 大');
  assertEq(kthLargest(arr, 2, numberAsc), 5, '第 2 大');
  assertEq(kthLargest(arr, 6, numberAsc), 1, '第 6 大');
});

test('kthSmallest: k 越界抛出错误', () => {
  const arr = [1, 2, 3];
  let threw = false;
  try {
    kthSmallest(arr, 0, numberAsc);
  } catch {
    threw = true;
  }
  assert(threw, 'k=0 应抛出错误');
});

// ============================================================================
// 中位数测试
// ============================================================================

console.log('\n--- 中位数测试 ---');

test('median: 奇数个元素', () => {
  const arr = [5, 3, 1, 4, 2];
  // 排序后 [1, 2, 3, 4, 5]，中位数是 3
  assertEq(median(arr), 3, '中位数应为 3');
});

test('median: 偶数个元素', () => {
  const arr = [4, 2, 1, 3];
  // 排序后 [1, 2, 3, 4]，中位数是 (2+3)/2 = 2.5
  assertEq(median(arr), 2.5, '中位数应为 2.5');
});

test('median: 单元素', () => {
  assertEq(median([42]), 42, '单元素中位数');
});

test('median: 两元素', () => {
  assertEq(median([1, 3]), 2, '两元素中位数');
});

test('median: 空数组抛出错误', () => {
  let threw = false;
  try {
    median([]);
  } catch {
    threw = true;
  }
  assert(threw, '空数组应抛出错误');
});

// ============================================================================
// 百分位数测试
// ============================================================================

console.log('\n--- 百分位数测试 ---');

test('percentile: P0 是最小值', () => {
  const arr = [1, 2, 3, 4, 5];
  assertEq(percentile(arr, 0), 1, 'P0 应为最小值');
});

test('percentile: P100 是最大值', () => {
  const arr = [1, 2, 3, 4, 5];
  assertEq(percentile(arr, 1), 5, 'P100 应为最大值');
});

test('percentile: P50 接近中位数', () => {
  const arr = [1, 2, 3, 4, 5];
  const p50 = percentile(arr, 0.5);
  assert(p50 >= 2 && p50 <= 4, 'P50 应接近中位数');
});

test('percentile: 边界检查', () => {
  const arr = [1, 2, 3];
  let threw = false;
  try {
    percentile(arr, 1.5);
  } catch {
    threw = true;
  }
  assert(threw, 'p > 1 应抛出错误');
});

// ============================================================================
// 大规模数据测试
// ============================================================================

console.log('\n--- 大规模数据测试 ---');

test('大规模随机数据', () => {
  const arr = generateNumbers(10000, 'random');
  const k = 500;

  // 快速选择的结果
  const result = kthSmallest(arr, k, numberAsc);

  // 对比排序后的结果
  const sorted = [...arr].sort((a, b) => a - b);
  assertEq(result, sorted[k - 1], '结果应与排序后一致');
});

test('大规模 TopK', () => {
  const arr = generateNumbers(10000, 'random');
  const k = 100;

  const topK = topKSmallestSorted(arr, k, numberAsc);
  const sorted = [...arr].sort((a, b) => a - b).slice(0, k);

  assert(
    JSON.stringify(topK) === JSON.stringify(sorted),
    'TopK 结果应与排序后一致'
  );
});

// ============================================================================
// 各种数据分布测试
// ============================================================================

console.log('\n--- 数据分布测试 ---');

const distributions = ['random', 'sorted', 'reversed', 'fewUnique'] as const;

for (const dist of distributions) {
  test(`分布: ${dist}`, () => {
    const arr = generateNumbers(1000, dist);
    const k = 100;

    const result = kthSmallest(arr, k, numberAsc);
    const sorted = [...arr].sort((a, b) => a - b);

    assertEq(result, sorted[k - 1], '结果应正确');
  });
}

// ============================================================================
// 元信息测试
// ============================================================================

console.log('\n--- 元信息测试 ---');

test('meta 信息正确', () => {
  assertEq(meta.timeComplexity.average, 'O(n)', '平均时间复杂度');
  assertEq(meta.timeComplexity.worst, 'O(n²)', '最坏时间复杂度');
  assertEq(meta.spaceComplexity, 'O(1)', '空间复杂度');
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

