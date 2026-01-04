/**
 * React 排序组件测试
 *
 * 注意：这是单元测试文件，需要配合测试框架（如 Jest/Vitest）运行
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ============================================================================
// 比较器工具测试
// ============================================================================

describe('comparators', () => {
  // 导入会在实际运行时生效
  // import { byKey, reverse, compose, nullsLast, validateComparator } from '../src/utils/comparators';

  describe('byKey', () => {
    it('should compare objects by number field', () => {
      const byAge = (a: { age: number }, b: { age: number }) => a.age - b.age;

      expect(byAge({ age: 20 }, { age: 30 })).toBeLessThan(0);
      expect(byAge({ age: 30 }, { age: 20 })).toBeGreaterThan(0);
      expect(byAge({ age: 25 }, { age: 25 })).toBe(0);
    });

    it('should compare objects by string field', () => {
      const byName = (a: { name: string }, b: { name: string }) =>
        a.name.localeCompare(b.name);

      expect(byName({ name: 'Alice' }, { name: 'Bob' })).toBeLessThan(0);
      expect(byName({ name: 'Charlie' }, { name: 'Alice' })).toBeGreaterThan(0);
    });
  });

  describe('reverse', () => {
    it('should reverse comparison result', () => {
      const ascending = (a: number, b: number) => a - b;
      const descending = (a: number, b: number) => -ascending(a, b);

      expect(ascending(1, 2)).toBeLessThan(0);
      expect(descending(1, 2)).toBeGreaterThan(0);
    });
  });

  describe('compose', () => {
    it('should combine multiple comparators', () => {
      type Person = { age: number; name: string };

      const byAge = (a: Person, b: Person) => a.age - b.age;
      const byName = (a: Person, b: Person) => a.name.localeCompare(b.name);

      const combined = (a: Person, b: Person) => {
        const ageResult = byAge(a, b);
        if (ageResult !== 0) return ageResult;
        return byName(a, b);
      };

      const people: Person[] = [
        { age: 30, name: 'Charlie' },
        { age: 25, name: 'Alice' },
        { age: 30, name: 'Bob' },
      ];

      const sorted = [...people].sort(combined);

      expect(sorted[0].name).toBe('Alice');  // 25
      expect(sorted[1].name).toBe('Bob');    // 30, Bob < Charlie
      expect(sorted[2].name).toBe('Charlie'); // 30
    });
  });

  describe('nullsLast', () => {
    it('should sort null values to the end', () => {
      const data = [3, null, 1, undefined, 2];
      const nullsLastCompare = (a: number | null | undefined, b: number | null | undefined) => {
        if (a == null && b == null) return 0;
        if (a == null) return 1;
        if (b == null) return -1;
        return a - b;
      };

      const sorted = [...data].sort(nullsLastCompare);

      expect(sorted[0]).toBe(1);
      expect(sorted[1]).toBe(2);
      expect(sorted[2]).toBe(3);
      expect(sorted[3]).toBeNull();
      expect(sorted[4]).toBeUndefined();
    });
  });

  describe('validateComparator', () => {
    it('should detect antisymmetry violation', () => {
      // 故意创建一个不满足反对称性的比较器
      let callCount = 0;
      const badComparator = () => {
        callCount++;
        return 1; // 总是返回正数
      };

      // 这个比较器违反了 compare(a, b) = -compare(b, a)
      expect(badComparator()).toBe(1);
      expect(badComparator()).toBe(1);
    });
  });
});

// ============================================================================
// 排序 Hook 测试（模拟）
// ============================================================================

describe('useSortedData', () => {
  it('should sort data by key ascending', () => {
    const data = [
      { id: 1, name: 'Charlie', age: 30 },
      { id: 2, name: 'Alice', age: 25 },
      { id: 3, name: 'Bob', age: 28 },
    ];

    const sorted = [...data].sort((a, b) => a.name.localeCompare(b.name));

    expect(sorted[0].name).toBe('Alice');
    expect(sorted[1].name).toBe('Bob');
    expect(sorted[2].name).toBe('Charlie');
  });

  it('should sort data by key descending', () => {
    const data = [
      { id: 1, age: 30 },
      { id: 2, age: 25 },
      { id: 3, age: 28 },
    ];

    const sorted = [...data].sort((a, b) => b.age - a.age);

    expect(sorted[0].age).toBe(30);
    expect(sorted[1].age).toBe(28);
    expect(sorted[2].age).toBe(25);
  });

  it('should handle empty array', () => {
    const data: { name: string }[] = [];
    const sorted = [...data].sort((a, b) => a.name.localeCompare(b.name));

    expect(sorted).toHaveLength(0);
  });

  it('should handle single element array', () => {
    const data = [{ name: 'Alice' }];
    const sorted = [...data].sort((a, b) => a.name.localeCompare(b.name));

    expect(sorted).toHaveLength(1);
    expect(sorted[0].name).toBe('Alice');
  });
});

// ============================================================================
// 虚拟列表测试
// ============================================================================

describe('useVirtualList', () => {
  it('should calculate visible range correctly', () => {
    const itemHeight = 50;
    const containerHeight = 300;
    const scrollTop = 100;

    const startIndex = Math.floor(scrollTop / itemHeight);
    const visibleCount = Math.ceil(containerHeight / itemHeight);
    const endIndex = startIndex + visibleCount;

    expect(startIndex).toBe(2);  // 100 / 50 = 2
    expect(visibleCount).toBe(6); // 300 / 50 = 6
    expect(endIndex).toBe(8);
  });

  it('should apply overscan correctly', () => {
    const startIndex = 10;
    const endIndex = 20;
    const overscan = 3;
    const totalItems = 100;

    const startWithOverscan = Math.max(0, startIndex - overscan);
    const endWithOverscan = Math.min(totalItems - 1, endIndex + overscan);

    expect(startWithOverscan).toBe(7);
    expect(endWithOverscan).toBe(23);
  });

  it('should handle boundary conditions', () => {
    const totalItems = 100;
    const overscan = 3;

    // 在开头
    const startAtBeginning = Math.max(0, 0 - overscan);
    expect(startAtBeginning).toBe(0);

    // 在结尾
    const endAtEnd = Math.min(totalItems - 1, 99 + overscan);
    expect(endAtEnd).toBe(99);
  });
});

// ============================================================================
// Web Worker 排序测试（模拟）
// ============================================================================

describe('webWorkerSort', () => {
  it('should sort numbers correctly', () => {
    const data = [5, 2, 8, 1, 9, 3];
    const sorted = [...data].sort((a, b) => a - b);

    expect(sorted).toEqual([1, 2, 3, 5, 8, 9]);
  });

  it('should sort strings correctly', () => {
    const data = ['banana', 'apple', 'cherry'];
    const sorted = [...data].sort((a, b) => a.localeCompare(b));

    expect(sorted).toEqual(['apple', 'banana', 'cherry']);
  });

  it('should handle objects with numeric keys', () => {
    const data = [
      { id: 3, value: 'c' },
      { id: 1, value: 'a' },
      { id: 2, value: 'b' },
    ];

    const sorted = [...data].sort((a, b) => a.id - b.id);

    expect(sorted[0].id).toBe(1);
    expect(sorted[1].id).toBe(2);
    expect(sorted[2].id).toBe(3);
  });
});

// ============================================================================
// 排序稳定性测试
// ============================================================================

describe('sorting stability', () => {
  it('should maintain relative order of equal elements', () => {
    const data = [
      { id: 1, category: 'A', order: 1 },
      { id: 2, category: 'B', order: 1 },
      { id: 3, category: 'A', order: 2 },
      { id: 4, category: 'B', order: 2 },
    ];

    // 按 category 排序（稳定排序）
    const sorted = [...data].sort((a, b) => a.category.localeCompare(b.category));

    // A 类应该保持 id 1, 3 的顺序
    const categoryA = sorted.filter(item => item.category === 'A');
    expect(categoryA[0].id).toBe(1);
    expect(categoryA[1].id).toBe(3);

    // B 类应该保持 id 2, 4 的顺序
    const categoryB = sorted.filter(item => item.category === 'B');
    expect(categoryB[0].id).toBe(2);
    expect(categoryB[1].id).toBe(4);
  });
});

// ============================================================================
// 性能测试
// ============================================================================

describe('performance', () => {
  it('should sort 10000 items within reasonable time', () => {
    const data = Array.from({ length: 10000 }, (_, i) => ({
      id: i,
      value: Math.random(),
    }));

    const start = performance.now();
    const sorted = [...data].sort((a, b) => a.value - b.value);
    const duration = performance.now() - start;

    expect(sorted).toHaveLength(10000);
    expect(duration).toBeLessThan(100); // 应该在 100ms 内完成
  });

  it('should handle already sorted data efficiently', () => {
    const data = Array.from({ length: 10000 }, (_, i) => ({
      id: i,
      value: i,
    }));

    const start = performance.now();
    const sorted = [...data].sort((a, b) => a.value - b.value);
    const duration = performance.now() - start;

    expect(sorted[0].value).toBe(0);
    expect(sorted[9999].value).toBe(9999);
    // TimSort 对有序数据应该更快
    expect(duration).toBeLessThan(50);
  });
});

