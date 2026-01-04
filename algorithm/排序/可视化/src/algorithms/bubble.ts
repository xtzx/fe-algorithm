/**
 * 冒泡排序可视化版本
 */

import type { SortStep, SortGenerator } from '../visualizer';

/**
 * 冒泡排序步骤生成器
 *
 * 特点：相邻元素比较，大的元素"冒泡"到末尾
 */
export function* bubbleSortSteps(arr: number[]): SortGenerator {
  const n = arr.length;

  for (let i = 0; i < n - 1; i++) {
    let swapped = false;

    for (let j = 0; j < n - i - 1; j++) {
      // 比较相邻元素
      yield { type: 'compare', indices: [j, j + 1] };

      if (arr[j] > arr[j + 1]) {
        // 交换
        yield { type: 'swap', indices: [j, j + 1] };
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        swapped = true;
      }
    }

    // 标记已排序位置
    yield { type: 'sorted', indices: [n - i - 1] };

    // 优化：如果没有交换，数组已有序
    if (!swapped) {
      // 标记剩余所有为已排序
      const remaining = [];
      for (let k = 0; k < n - i - 1; k++) {
        remaining.push(k);
      }
      if (remaining.length > 0) {
        yield { type: 'sorted', indices: remaining };
      }
      break;
    }
  }

  // 确保第一个元素也标记为已排序
  yield { type: 'sorted', indices: [0] };
}

/**
 * 鸡尾酒排序（双向冒泡）步骤生成器
 */
export function* cocktailSortSteps(arr: number[]): SortGenerator {
  const n = arr.length;
  let left = 0;
  let right = n - 1;
  let swapped = true;

  while (swapped && left < right) {
    swapped = false;

    // 从左到右
    for (let i = left; i < right; i++) {
      yield { type: 'compare', indices: [i, i + 1] };

      if (arr[i] > arr[i + 1]) {
        yield { type: 'swap', indices: [i, i + 1] };
        [arr[i], arr[i + 1]] = [arr[i + 1], arr[i]];
        swapped = true;
      }
    }

    yield { type: 'sorted', indices: [right] };
    right--;

    if (!swapped) break;
    swapped = false;

    // 从右到左
    for (let i = right; i > left; i--) {
      yield { type: 'compare', indices: [i - 1, i] };

      if (arr[i - 1] > arr[i]) {
        yield { type: 'swap', indices: [i - 1, i] };
        [arr[i - 1], arr[i]] = [arr[i], arr[i - 1]];
        swapped = true;
      }
    }

    yield { type: 'sorted', indices: [left] };
    left++;
  }

  // 标记剩余为已排序
  const remaining = [];
  for (let i = left; i <= right; i++) {
    remaining.push(i);
  }
  if (remaining.length > 0) {
    yield { type: 'sorted', indices: remaining };
  }
}

