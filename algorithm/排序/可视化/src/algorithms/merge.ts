/**
 * 归并排序可视化版本
 */

import type { SortStep, SortGenerator } from '../visualizer';

/**
 * 归并排序步骤生成器
 */
export function* mergeSortSteps(arr: number[]): SortGenerator {
  const temp = new Array(arr.length);
  yield* mergeSortHelper(arr, temp, 0, arr.length - 1);
}

function* mergeSortHelper(
  arr: number[],
  temp: number[],
  left: number,
  right: number
): SortGenerator {
  if (left >= right) {
    if (left === right) {
      yield { type: 'sorted', indices: [left] };
    }
    return;
  }

  const mid = Math.floor((left + right) / 2);

  // 递归排序左半部分
  yield* mergeSortHelper(arr, temp, left, mid);

  // 递归排序右半部分
  yield* mergeSortHelper(arr, temp, mid + 1, right);

  // 合并
  yield* merge(arr, temp, left, mid, right);
}

/**
 * 合并过程
 */
function* merge(
  arr: number[],
  temp: number[],
  left: number,
  mid: number,
  right: number
): SortGenerator {
  // 复制到临时数组
  for (let i = left; i <= right; i++) {
    temp[i] = arr[i];
  }

  let i = left;      // 左半部分指针
  let j = mid + 1;   // 右半部分指针
  let k = left;      // 合并结果指针

  while (i <= mid && j <= right) {
    // 比较
    yield { type: 'compare', indices: [i, j] };

    if (temp[i] <= temp[j]) {
      // 取左边的元素
      if (arr[k] !== temp[i]) {
        yield { type: 'highlight', indices: [k] };
      }
      arr[k] = temp[i];
      i++;
    } else {
      // 取右边的元素
      yield { type: 'highlight', indices: [k] };
      arr[k] = temp[j];
      j++;
    }
    k++;
  }

  // 复制剩余的左半部分
  while (i <= mid) {
    yield { type: 'highlight', indices: [k] };
    arr[k] = temp[i];
    i++;
    k++;
  }

  // 复制剩余的右半部分
  while (j <= right) {
    yield { type: 'highlight', indices: [k] };
    arr[k] = temp[j];
    j++;
    k++;
  }

  // 标记合并区域为已排序（如果是最终合并）
  if (left === 0 && right === arr.length - 1) {
    const indices = [];
    for (let x = left; x <= right; x++) {
      indices.push(x);
    }
    yield { type: 'sorted', indices };
  }
}

/**
 * 自底向上归并排序步骤生成器
 */
export function* mergeSortBottomUpSteps(arr: number[]): SortGenerator {
  const n = arr.length;
  const temp = new Array(n);

  // 子数组大小从 1 开始，每次翻倍
  for (let size = 1; size < n; size *= 2) {
    // 合并相邻的子数组
    for (let left = 0; left < n - size; left += size * 2) {
      const mid = left + size - 1;
      const right = Math.min(left + size * 2 - 1, n - 1);

      yield* merge(arr, temp, left, mid, right);
    }
  }

  // 标记所有为已排序
  const indices = [];
  for (let i = 0; i < n; i++) {
    indices.push(i);
  }
  yield { type: 'sorted', indices };
}

