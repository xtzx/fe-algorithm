/**
 * SortedWindow - 有序窗口
 *
 * 维护固定容量的有序窗口，支持 Top K 场景
 */

export type Comparator<T> = (a: T, b: T) => number;

/**
 * 有序窗口
 *
 * 维护最多 capacity 个元素的有序集合。
 * 当窗口满时，新元素如果比最小元素大，则替换最小元素。
 *
 * @example
 * // 维护 Top 5 最大值
 * const window = new SortedWindow<number>(5, (a, b) => a - b);
 * window.add(10);
 * window.add(3);
 * window.add(7);
 * console.log(window.toArray()); // [3, 7, 10]
 */
export class SortedWindow<T> {
  private items: T[] = [];
  private readonly capacity: number;
  private readonly cmp: Comparator<T>;

  constructor(capacity: number, cmp: Comparator<T>) {
    if (capacity <= 0) {
      throw new Error('Capacity must be positive');
    }
    this.capacity = capacity;
    this.cmp = cmp;
  }

  /**
   * 添加元素
   *
   * 时间复杂度: O(k)
   * - O(log k) 二分查找
   * - O(k) 插入/删除
   */
  add(item: T): void {
    const pos = this.findInsertPosition(item);

    // 窗口未满
    if (this.items.length < this.capacity) {
      this.items.splice(pos, 0, item);
      return;
    }

    // 窗口已满
    // 如果新元素比最小的还小（或等于），不插入
    if (pos === 0 && this.cmp(item, this.items[0]) <= 0) {
      return;
    }

    // 移除最小，插入新元素
    // 注意：插入位置需要调整（因为删除了一个元素）
    if (pos > 0) {
      this.items.shift();
      this.items.splice(pos - 1, 0, item);
    }
  }

  /**
   * 二分查找插入位置
   */
  private findInsertPosition(item: T): number {
    let lo = 0;
    let hi = this.items.length;

    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.cmp(this.items[mid], item) < 0) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    return lo;
  }

  /**
   * 获取第 k 个元素（0-indexed）
   *
   * 时间复杂度: O(1)
   */
  get(k: number): T | undefined {
    if (k < 0 || k >= this.items.length) {
      return undefined;
    }
    return this.items[k];
  }

  /**
   * 获取最小元素（第一个）
   */
  getMin(): T | undefined {
    return this.items[0];
  }

  /**
   * 获取最大元素（最后一个）
   */
  getMax(): T | undefined {
    return this.items[this.items.length - 1];
  }

  /**
   * 转换为数组
   *
   * 时间复杂度: O(k)
   */
  toArray(): T[] {
    return [...this.items];
  }

  /**
   * 当前大小
   */
  get size(): number {
    return this.items.length;
  }

  /**
   * 是否为空
   */
  get isEmpty(): boolean {
    return this.items.length === 0;
  }

  /**
   * 是否已满
   */
  get isFull(): boolean {
    return this.items.length >= this.capacity;
  }

  /**
   * 清空窗口
   */
  clear(): void {
    this.items = [];
  }

  /**
   * 检查元素是否存在
   */
  contains(item: T): boolean {
    const pos = this.findInsertPosition(item);
    return pos < this.items.length && this.cmp(this.items[pos], item) === 0;
  }

  /**
   * 迭代器支持
   */
  *[Symbol.iterator](): Iterator<T> {
    for (const item of this.items) {
      yield item;
    }
  }
}

// ============================================================================
// 便捷类型
// ============================================================================

/**
 * 数值型有序窗口
 */
export class NumberWindow extends SortedWindow<number> {
  constructor(capacity: number, ascending = true) {
    super(capacity, ascending ? (a, b) => a - b : (a, b) => b - a);
  }
}

/**
 * 字符串有序窗口
 */
export class StringWindow extends SortedWindow<string> {
  constructor(capacity: number, ascending = true) {
    super(
      capacity,
      ascending
        ? (a, b) => a.localeCompare(b)
        : (a, b) => b.localeCompare(a)
    );
  }
}

