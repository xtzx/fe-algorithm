/**
 * 流式排序
 *
 * 导出所有流式排序相关的数据结构
 */

export { SortedWindow } from './sortedWindow';
export { OnlineMedian } from './onlineMedian';
export { MinHeap, MaxHeap } from './onlineMedian';

// ============================================================================
// 便捷工厂函数
// ============================================================================

import { SortedWindow } from './sortedWindow';
import { OnlineMedian } from './onlineMedian';

/**
 * 创建 Top K 最大值窗口
 */
export function createTopKMax<T>(
  k: number,
  cmp: (a: T, b: T) => number
): SortedWindow<T> {
  return new SortedWindow(k, cmp);
}

/**
 * 创建 Top K 最小值窗口
 */
export function createTopKMin<T>(
  k: number,
  cmp: (a: T, b: T) => number
): SortedWindow<T> {
  return new SortedWindow(k, (a, b) => -cmp(a, b));
}

/**
 * 创建在线中位数追踪器
 */
export function createMedianTracker(): OnlineMedian {
  return new OnlineMedian();
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '流式排序',
  structures: {
    SortedWindow: {
      description: '维护固定容量的有序窗口',
      insertComplexity: 'O(k)',
      queryComplexity: 'O(1)',
      spaceComplexity: 'O(k)',
    },
    OnlineMedian: {
      description: '实时计算数据流中位数',
      insertComplexity: 'O(log n)',
      queryComplexity: 'O(1)',
      spaceComplexity: 'O(n)',
    },
  },
  relatedProblems: [
    'LeetCode 295: 数据流的中位数',
    'LeetCode 703: 数据流中的第K大元素',
    'LeetCode 480: 滑动窗口中位数',
  ],
};

