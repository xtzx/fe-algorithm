/**
 * 快速排序可视化版本
 */

import type { SortStep, SortGenerator } from '../visualizer';

/**
 * 快速排序步骤生成器
 */
export function* quickSortSteps(arr: number[]): SortGenerator {
  yield* quickSortHelper(arr, 0, arr.length - 1);
}

function* quickSortHelper(
  arr: number[],
  low: number,
  high: number
): SortGenerator {
  if (low >= high) {
    if (low === high) {
      yield { type: 'sorted', indices: [low] };
    }
    return;
  }

  // 分区
  const pivotIndex = yield* partition(arr, low, high);

  // 标记 pivot 位置为已排序
  yield { type: 'sorted', indices: [pivotIndex] };

  // 递归排序左右两边
  yield* quickSortHelper(arr, low, pivotIndex - 1);
  yield* quickSortHelper(arr, pivotIndex + 1, high);
}

/**
 * 分区过程
 */
function* partition(
  arr: number[],
  low: number,
  high: number
): Generator<SortStep, number, unknown> {
  // 使用三数取中选择 pivot
  const mid = Math.floor((low + high) / 2);

  // 将 mid 元素移到末尾作为 pivot
  if (mid !== high) {
    yield { type: 'compare', indices: [mid, high] };
    yield { type: 'swap', indices: [mid, high] };
    [arr[mid], arr[high]] = [arr[high], arr[mid]];
  }

  const pivot = arr[high];

  // 标记 pivot
  yield { type: 'pivot', indices: [high] };

  let i = low - 1;

  for (let j = low; j < high; j++) {
    // 比较
    yield { type: 'compare', indices: [j, high] };

    if (arr[j] <= pivot) {
      i++;
      if (i !== j) {
        yield { type: 'swap', indices: [i, j] };
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
    }
  }

  // 将 pivot 放到正确位置
  const pivotPos = i + 1;
  if (pivotPos !== high) {
    yield { type: 'swap', indices: [pivotPos, high] };
    [arr[pivotPos], arr[high]] = [arr[high], arr[pivotPos]];
  }

  return pivotPos;
}

/**
 * 三路快排步骤生成器（处理大量重复）
 */
export function* quickSort3WaySteps(arr: number[]): SortGenerator {
  yield* quickSort3WayHelper(arr, 0, arr.length - 1);
}

function* quickSort3WayHelper(
  arr: number[],
  low: number,
  high: number
): SortGenerator {
  if (low >= high) {
    if (low === high) {
      yield { type: 'sorted', indices: [low] };
    }
    return;
  }

  // 三路分区
  const pivot = arr[low];
  yield { type: 'pivot', indices: [low] };

  let lt = low;      // arr[low..lt-1] < pivot
  let i = low + 1;   // arr[lt..i-1] == pivot
  let gt = high;     // arr[gt+1..high] > pivot

  while (i <= gt) {
    yield { type: 'compare', indices: [i, low] };

    if (arr[i] < pivot) {
      yield { type: 'swap', indices: [lt, i] };
      [arr[lt], arr[i]] = [arr[i], arr[lt]];
      lt++;
      i++;
    } else if (arr[i] > pivot) {
      yield { type: 'swap', indices: [i, gt] };
      [arr[i], arr[gt]] = [arr[gt], arr[i]];
      gt--;
    } else {
      i++;
    }
  }

  // 标记等于 pivot 的区域为已排序
  const equalRange = [];
  for (let k = lt; k <= gt; k++) {
    equalRange.push(k);
  }
  yield { type: 'sorted', indices: equalRange };

  // 递归
  yield* quickSort3WayHelper(arr, low, lt - 1);
  yield* quickSort3WayHelper(arr, gt + 1, high);
}

