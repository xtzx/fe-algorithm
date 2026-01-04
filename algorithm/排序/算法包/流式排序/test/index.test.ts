/**
 * 流式排序测试
 */

import { describe, it, expect } from 'vitest';
import { SortedWindow, NumberWindow, StringWindow } from '../src/sortedWindow';
import { OnlineMedian, OnlineMedianWithRemoval, MinHeap, MaxHeap } from '../src/onlineMedian';

// ============================================================================
// 堆测试
// ============================================================================

describe('MinHeap', () => {
  it('should create empty heap', () => {
    const heap = new MinHeap<number>();
    expect(heap.isEmpty()).toBe(true);
    expect(heap.size()).toBe(0);
  });

  it('should push and pop elements', () => {
    const heap = new MinHeap<number>();
    heap.push(3);
    heap.push(1);
    heap.push(2);

    expect(heap.pop()).toBe(1);
    expect(heap.pop()).toBe(2);
    expect(heap.pop()).toBe(3);
  });

  it('should peek without removing', () => {
    const heap = new MinHeap<number>();
    heap.push(5);
    heap.push(3);

    expect(heap.peek()).toBe(3);
    expect(heap.size()).toBe(2);
  });

  it('should handle duplicates', () => {
    const heap = new MinHeap<number>();
    heap.push(2);
    heap.push(2);
    heap.push(1);
    heap.push(2);

    expect(heap.pop()).toBe(1);
    expect(heap.pop()).toBe(2);
    expect(heap.pop()).toBe(2);
    expect(heap.pop()).toBe(2);
  });
});

describe('MaxHeap', () => {
  it('should return max element first', () => {
    const heap = new MaxHeap<number>();
    heap.push(1);
    heap.push(3);
    heap.push(2);

    expect(heap.pop()).toBe(3);
    expect(heap.pop()).toBe(2);
    expect(heap.pop()).toBe(1);
  });
});

// ============================================================================
// SortedWindow 测试
// ============================================================================

describe('SortedWindow', () => {
  describe('basic operations', () => {
    it('should create empty window', () => {
      const window = new SortedWindow<number>(5, (a, b) => a - b);
      expect(window.isEmpty).toBe(true);
      expect(window.size).toBe(0);
    });

    it('should add elements in sorted order', () => {
      const window = new SortedWindow<number>(5, (a, b) => a - b);
      window.add(3);
      window.add(1);
      window.add(2);

      expect(window.toArray()).toEqual([1, 2, 3]);
    });

    it('should respect capacity', () => {
      const window = new SortedWindow<number>(3, (a, b) => a - b);
      window.add(5);
      window.add(3);
      window.add(7);
      window.add(2);
      window.add(8);

      // 保留最大的 3 个：5, 7, 8
      expect(window.size).toBe(3);
      expect(window.toArray()).toEqual([5, 7, 8]);
    });

    it('should not add elements smaller than minimum when full', () => {
      const window = new SortedWindow<number>(3, (a, b) => a - b);
      window.add(5);
      window.add(6);
      window.add(7);
      window.add(1); // 比最小的 5 还小，不应该加入

      expect(window.toArray()).toEqual([5, 6, 7]);
    });
  });

  describe('get methods', () => {
    it('should get element by index', () => {
      const window = new SortedWindow<number>(5, (a, b) => a - b);
      window.add(3);
      window.add(1);
      window.add(2);

      expect(window.get(0)).toBe(1);
      expect(window.get(1)).toBe(2);
      expect(window.get(2)).toBe(3);
      expect(window.get(3)).toBeUndefined();
      expect(window.get(-1)).toBeUndefined();
    });

    it('should get min and max', () => {
      const window = new SortedWindow<number>(5, (a, b) => a - b);
      window.add(3);
      window.add(1);
      window.add(5);

      expect(window.getMin()).toBe(1);
      expect(window.getMax()).toBe(5);
    });
  });

  describe('utility methods', () => {
    it('should check if contains element', () => {
      const window = new SortedWindow<number>(5, (a, b) => a - b);
      window.add(1);
      window.add(3);
      window.add(5);

      expect(window.contains(3)).toBe(true);
      expect(window.contains(2)).toBe(false);
    });

    it('should be iterable', () => {
      const window = new SortedWindow<number>(5, (a, b) => a - b);
      window.add(3);
      window.add(1);
      window.add(2);

      const items = [...window];
      expect(items).toEqual([1, 2, 3]);
    });

    it('should clear', () => {
      const window = new SortedWindow<number>(5, (a, b) => a - b);
      window.add(1);
      window.add(2);
      window.clear();

      expect(window.isEmpty).toBe(true);
    });
  });

  describe('with objects', () => {
    interface Item {
      id: number;
      value: string;
    }

    it('should sort objects by comparator', () => {
      const window = new SortedWindow<Item>(3, (a, b) => a.id - b.id);
      window.add({ id: 3, value: 'c' });
      window.add({ id: 1, value: 'a' });
      window.add({ id: 2, value: 'b' });

      expect(window.toArray().map(x => x.id)).toEqual([1, 2, 3]);
    });
  });
});

describe('NumberWindow', () => {
  it('should sort numbers ascending by default', () => {
    const window = new NumberWindow(3);
    window.add(3);
    window.add(1);
    window.add(2);

    expect(window.toArray()).toEqual([1, 2, 3]);
  });

  it('should sort numbers descending when specified', () => {
    const window = new NumberWindow(3, false);
    window.add(3);
    window.add(1);
    window.add(2);

    expect(window.toArray()).toEqual([3, 2, 1]);
  });
});

describe('StringWindow', () => {
  it('should sort strings', () => {
    const window = new StringWindow(3);
    window.add('banana');
    window.add('apple');
    window.add('cherry');

    expect(window.toArray()).toEqual(['apple', 'banana', 'cherry']);
  });
});

// ============================================================================
// OnlineMedian 测试
// ============================================================================

describe('OnlineMedian', () => {
  describe('basic operations', () => {
    it('should return single element as median', () => {
      const median = new OnlineMedian();
      median.add(5);
      expect(median.getMedian()).toBe(5);
    });

    it('should return average of two elements', () => {
      const median = new OnlineMedian();
      median.add(1);
      median.add(2);
      expect(median.getMedian()).toBe(1.5);
    });

    it('should return middle element for odd count', () => {
      const median = new OnlineMedian();
      median.add(1);
      median.add(2);
      median.add(3);
      expect(median.getMedian()).toBe(2);
    });

    it('should throw on empty', () => {
      const median = new OnlineMedian();
      expect(() => median.getMedian()).toThrow();
    });
  });

  describe('complex cases', () => {
    it('should handle LeetCode example 1', () => {
      const median = new OnlineMedian();
      median.add(1);
      median.add(2);
      expect(median.getMedian()).toBe(1.5);
      median.add(3);
      expect(median.getMedian()).toBe(2);
    });

    it('should handle unsorted input', () => {
      const median = new OnlineMedian();
      median.add(5);
      median.add(2);
      median.add(8);
      median.add(1);
      median.add(9);

      // 排序后: 1, 2, 5, 8, 9
      expect(median.getMedian()).toBe(5);
    });

    it('should handle duplicates', () => {
      const median = new OnlineMedian();
      median.add(1);
      median.add(1);
      median.add(1);
      expect(median.getMedian()).toBe(1);
    });

    it('should handle negative numbers', () => {
      const median = new OnlineMedian();
      median.add(-1);
      median.add(-5);
      median.add(0);
      median.add(3);

      // 排序后: -5, -1, 0, 3
      expect(median.getMedian()).toBe(-0.5);
    });

    it('should handle large sequence', () => {
      const median = new OnlineMedian();

      // 添加 1-100
      for (let i = 1; i <= 100; i++) {
        median.add(i);
      }

      // 中位数是 50 和 51 的平均值
      expect(median.getMedian()).toBe(50.5);
    });

    it('should track count', () => {
      const median = new OnlineMedian();
      median.add(1);
      median.add(2);
      median.add(3);
      expect(median.count).toBe(3);
    });
  });
});

// ============================================================================
// OnlineMedianWithRemoval 测试
// ============================================================================

describe('OnlineMedianWithRemoval', () => {
  it('should work like basic OnlineMedian', () => {
    const median = new OnlineMedianWithRemoval();
    median.add(1);
    median.add(2);
    median.add(3);
    expect(median.getMedian()).toBe(2);
  });

  it('should handle removal', () => {
    const median = new OnlineMedianWithRemoval();
    median.add(1);
    median.add(2);
    median.add(3);

    median.remove(1);
    // 剩余: 2, 3
    expect(median.getMedian()).toBe(2.5);
  });

  it('should handle multiple removals', () => {
    const median = new OnlineMedianWithRemoval();
    median.add(1);
    median.add(2);
    median.add(3);
    median.add(4);
    median.add(5);

    // 移除 2 和 4
    median.remove(2);
    median.remove(4);

    // 剩余: 1, 3, 5
    expect(median.getMedian()).toBe(3);
  });
});

// ============================================================================
// 集成测试
// ============================================================================

describe('Integration', () => {
  it('should work with streaming data simulation', () => {
    const window = new SortedWindow<{ timestamp: number; value: number }>(
      5,
      (a, b) => b.value - a.value // 按 value 降序
    );

    // 模拟流式数据
    const stream = [
      { timestamp: 1, value: 100 },
      { timestamp: 2, value: 150 },
      { timestamp: 3, value: 80 },
      { timestamp: 4, value: 200 },
      { timestamp: 5, value: 120 },
      { timestamp: 6, value: 180 },
      { timestamp: 7, value: 90 },
    ];

    for (const item of stream) {
      window.add(item);
    }

    // Top 5 highest values: 200, 180, 150, 120, 100
    const values = window.toArray().map(x => x.value);
    expect(values).toEqual([200, 180, 150, 120, 100]);
  });

  it('should simulate stock price monitoring', () => {
    const median = new OnlineMedian();
    const prices = [100, 102, 99, 105, 98, 110, 95, 108];

    for (const price of prices) {
      median.add(price);
    }

    // 排序后: 95, 98, 99, 100, 102, 105, 108, 110
    // 中位数: (100 + 102) / 2 = 101
    expect(median.getMedian()).toBe(101);
  });
});

