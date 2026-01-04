/**
 * 堆排序 - 测试用例
 */

import {
  sort,
  sortInPlace,
  sortWithStats,
  buildMaxHeap,
  heapifyDown,
  findTopKLargest,
  findTopKSmallest,
  MaxPriorityQueue,
  MinPriorityQueue,
  meta,
} from '../src/index';
import { verifySort } from '../../公共库/src/正确性校验';
import { numberAsc, byField } from '../../公共库/src/比较器';
import {
  generateNumbers,
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
    const sorted = sort(data, numberAsc);
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
// 建堆测试
// ============================================================================

console.log('\n--- 建堆测试 ---');

test('buildMaxHeap 建立最大堆', () => {
  const arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
  buildMaxHeap(arr, numberAsc);

  // 验证堆性质：每个父节点 >= 子节点
  for (let i = 0; i < arr.length; i++) {
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    if (left < arr.length) {
      assert(arr[i] >= arr[left], `arr[${i}]=${arr[i]} 应 >= arr[${left}]=${arr[left]}`);
    }
    if (right < arr.length) {
      assert(arr[i] >= arr[right], `arr[${i}]=${arr[i]} 应 >= arr[${right}]=${arr[right]}`);
    }
  }
});

test('heapifyDown 向下堆化', () => {
  const arr = [1, 10, 5, 8, 7]; // 根节点不满足堆性质
  heapifyDown(arr, arr.length, 0, numberAsc);
  // 堆化后根应该是最大的
  assert(arr[0] === 10, `根应为最大值 10，实际为 ${arr[0]}`);
});

// ============================================================================
// TopK 测试
// ============================================================================

console.log('\n--- TopK 测试 ---');

test('findTopKLargest 找最大的 K 个', () => {
  const arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
  const top3 = findTopKLargest(arr, 3, numberAsc);

  // 验证结果包含 9, 6, 5
  const sorted = top3.sort((a, b) => b - a);
  assert(sorted[0] === 9, '第一大应为 9');
  assert(sorted[1] === 6, '第二大应为 6');
  assert(sorted[2] === 5, '第三大应为 5');
});

test('findTopKSmallest 找最小的 K 个', () => {
  const arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
  const bottom3 = findTopKSmallest(arr, 3, numberAsc);

  const sorted = bottom3.sort((a, b) => a - b);
  assert(sorted[0] === 1, '第一小应为 1');
  assert(sorted[1] === 1, '第二小应为 1');
  assert(sorted[2] === 2, '第三小应为 2');
});

test('TopK 边界情况: k=0', () => {
  const arr = [3, 1, 4];
  const result = findTopKLargest(arr, 0, numberAsc);
  assert(result.length === 0, 'k=0 应返回空数组');
});

test('TopK 边界情况: k >= n', () => {
  const arr = [3, 1, 4];
  const result = findTopKLargest(arr, 5, numberAsc);
  assert(result.length === 3, 'k>=n 应返回全部元素');
});

// ============================================================================
// 优先队列测试
// ============================================================================

console.log('\n--- 优先队列测试 ---');

test('MaxPriorityQueue 基本操作', () => {
  const pq = new MaxPriorityQueue<number>(numberAsc);

  pq.push(3);
  pq.push(1);
  pq.push(4);
  pq.push(1);
  pq.push(5);

  assert(pq.size === 5, '大小应为 5');
  assert(pq.peek() === 5, '堆顶应为 5');

  const pops = [];
  while (!pq.isEmpty()) {
    pops.push(pq.pop());
  }

  assert(
    JSON.stringify(pops) === JSON.stringify([5, 4, 3, 1, 1]),
    '弹出顺序应为 [5, 4, 3, 1, 1]'
  );
});

test('MinPriorityQueue 基本操作', () => {
  const pq = new MinPriorityQueue<number>(numberAsc);

  pq.push(3);
  pq.push(1);
  pq.push(4);
  pq.push(1);
  pq.push(5);

  assert(pq.peek() === 1, '堆顶应为 1');

  const pops = [];
  while (!pq.isEmpty()) {
    pops.push(pq.pop());
  }

  assert(
    JSON.stringify(pops) === JSON.stringify([1, 1, 3, 4, 5]),
    '弹出顺序应为 [1, 1, 3, 4, 5]'
  );
});

test('优先队列空操作', () => {
  const pq = new MaxPriorityQueue<number>(numberAsc);
  assert(pq.isEmpty(), '初始应为空');
  assert(pq.peek() === undefined, 'peek 空队列返回 undefined');
  assert(pq.pop() === undefined, 'pop 空队列返回 undefined');
});

// ============================================================================
// 统计测试
// ============================================================================

console.log('\n--- 统计测试 ---');

test('sortWithStats 返回正确统计', () => {
  const data = [5, 3, 8, 4, 2];
  const { result, comparisons, swaps } = sortWithStats(data, numberAsc);

  assert(
    JSON.stringify(result) === JSON.stringify([2, 3, 4, 5, 8]),
    '排序结果错误'
  );
  assert(comparisons > 0, '比较次数应大于 0');
  assert(swaps > 0, '交换次数应大于 0');
});

// ============================================================================
// 性能稳定性测试
// ============================================================================

console.log('\n--- 性能稳定性测试 ---');

test('有序数据不会退化', () => {
  // 对于快排，有序数据是最坏情况；但堆排不会
  const sortedData = generateNumbers(5000, 'sorted');
  const original = [...sortedData];

  const start = performance.now();
  const result = sort(sortedData, numberAsc);
  const time = performance.now() - start;

  const verification = verifySort(original, result, numberAsc, false);
  assert(verification.passed, '排序应正确');
  console.log(`   有序数据排序时间: ${time.toFixed(2)}ms`);
});

// ============================================================================
// 元信息测试
// ============================================================================

console.log('\n--- 元信息测试 ---');

test('meta 信息正确', () => {
  assert(meta.stable === false, '应为不稳定排序');
  assert(meta.inPlace === true, '应为原地排序');
  assert(meta.timeComplexity.worst === 'O(n log n)', '最坏时间复杂度保证');
  assert(meta.spaceComplexity === 'O(1)', '空间复杂度');
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

