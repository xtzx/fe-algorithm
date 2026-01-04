/**
 * 堆排序可视化版本
 */

import type { SortStep, SortGenerator } from '../visualizer';

/**
 * 堆排序步骤生成器
 */
export function* heapSortSteps(arr: number[]): SortGenerator {
  const n = arr.length;

  // 1. 建堆（从最后一个非叶子节点开始）
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    yield* heapify(arr, n, i);
  }

  // 2. 逐个取出堆顶
  for (let i = n - 1; i > 0; i--) {
    // 堆顶（最大）与末尾交换
    yield { type: 'compare', indices: [0, i] };
    yield { type: 'swap', indices: [0, i] };
    [arr[0], arr[i]] = [arr[i], arr[0]];

    // 标记为已排序
    yield { type: 'sorted', indices: [i] };

    // 重新堆化
    yield* heapify(arr, i, 0);
  }

  // 标记第一个元素为已排序
  yield { type: 'sorted', indices: [0] };
}

/**
 * 堆化过程
 *
 * @param arr 数组
 * @param n 堆的大小
 * @param i 要堆化的节点索引
 */
function* heapify(
  arr: number[],
  n: number,
  i: number
): SortGenerator {
  let largest = i;
  const left = 2 * i + 1;
  const right = 2 * i + 2;

  // 比较左子节点
  if (left < n) {
    yield { type: 'compare', indices: [left, largest] };
    if (arr[left] > arr[largest]) {
      largest = left;
    }
  }

  // 比较右子节点
  if (right < n) {
    yield { type: 'compare', indices: [right, largest] };
    if (arr[right] > arr[largest]) {
      largest = right;
    }
  }

  // 如果最大值不是当前节点
  if (largest !== i) {
    yield { type: 'swap', indices: [i, largest] };
    [arr[i], arr[largest]] = [arr[largest], arr[i]];

    // 递归堆化受影响的子树
    yield* heapify(arr, n, largest);
  }
}

/**
 * 小顶堆排序（降序）
 */
export function* heapSortDescSteps(arr: number[]): SortGenerator {
  const n = arr.length;

  // 建小顶堆
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    yield* minHeapify(arr, n, i);
  }

  // 逐个取出堆顶（最小）
  for (let i = n - 1; i > 0; i--) {
    yield { type: 'compare', indices: [0, i] };
    yield { type: 'swap', indices: [0, i] };
    [arr[0], arr[i]] = [arr[i], arr[0]];

    yield { type: 'sorted', indices: [i] };
    yield* minHeapify(arr, i, 0);
  }

  yield { type: 'sorted', indices: [0] };
}

function* minHeapify(
  arr: number[],
  n: number,
  i: number
): SortGenerator {
  let smallest = i;
  const left = 2 * i + 1;
  const right = 2 * i + 2;

  if (left < n) {
    yield { type: 'compare', indices: [left, smallest] };
    if (arr[left] < arr[smallest]) {
      smallest = left;
    }
  }

  if (right < n) {
    yield { type: 'compare', indices: [right, smallest] };
    if (arr[right] < arr[smallest]) {
      smallest = right;
    }
  }

  if (smallest !== i) {
    yield { type: 'swap', indices: [i, smallest] };
    [arr[i], arr[smallest]] = [arr[smallest], arr[i]];
    yield* minHeapify(arr, n, smallest);
  }
}

