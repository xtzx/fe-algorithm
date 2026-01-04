/**
 * OnlineMedian - 在线中位数
 *
 * 使用双堆（大顶堆 + 小顶堆）维护数据流的中位数
 *
 * LeetCode 295: 数据流的中位数
 */

// ============================================================================
// 堆实现
// ============================================================================

type HeapComparator<T> = (a: T, b: T) => number;

/**
 * 通用堆实现
 */
class Heap<T> {
  private data: T[] = [];
  private readonly cmp: HeapComparator<T>;

  constructor(cmp: HeapComparator<T>) {
    this.cmp = cmp;
  }

  /**
   * 插入元素
   */
  push(value: T): void {
    this.data.push(value);
    this.bubbleUp(this.data.length - 1);
  }

  /**
   * 弹出堆顶
   */
  pop(): T | undefined {
    if (this.data.length === 0) return undefined;

    const top = this.data[0];
    const last = this.data.pop()!;

    if (this.data.length > 0) {
      this.data[0] = last;
      this.bubbleDown(0);
    }

    return top;
  }

  /**
   * 查看堆顶
   */
  peek(): T | undefined {
    return this.data[0];
  }

  /**
   * 堆大小
   */
  size(): number {
    return this.data.length;
  }

  /**
   * 是否为空
   */
  isEmpty(): boolean {
    return this.data.length === 0;
  }

  /**
   * 上浮
   */
  private bubbleUp(index: number): void {
    while (index > 0) {
      const parentIndex = (index - 1) >>> 1;
      if (this.cmp(this.data[index], this.data[parentIndex]) >= 0) {
        break;
      }
      this.swap(index, parentIndex);
      index = parentIndex;
    }
  }

  /**
   * 下沉
   */
  private bubbleDown(index: number): void {
    const length = this.data.length;

    while (true) {
      let smallest = index;
      const leftChild = 2 * index + 1;
      const rightChild = 2 * index + 2;

      if (
        leftChild < length &&
        this.cmp(this.data[leftChild], this.data[smallest]) < 0
      ) {
        smallest = leftChild;
      }

      if (
        rightChild < length &&
        this.cmp(this.data[rightChild], this.data[smallest]) < 0
      ) {
        smallest = rightChild;
      }

      if (smallest === index) break;

      this.swap(index, smallest);
      index = smallest;
    }
  }

  /**
   * 交换元素
   */
  private swap(i: number, j: number): void {
    [this.data[i], this.data[j]] = [this.data[j], this.data[i]];
  }
}

/**
 * 最小堆
 */
export class MinHeap<T> extends Heap<T> {
  constructor(cmp: HeapComparator<T> = (a, b) => (a as number) - (b as number)) {
    super(cmp);
  }
}

/**
 * 最大堆
 */
export class MaxHeap<T> extends Heap<T> {
  constructor(cmp: HeapComparator<T> = (a, b) => (a as number) - (b as number)) {
    super((a, b) => -cmp(a, b)); // 反转比较器
  }
}

// ============================================================================
// 在线中位数
// ============================================================================

/**
 * 在线中位数
 *
 * 使用双堆维护数据的两半：
 * - maxHeap: 存储较小的一半，堆顶是其中最大的
 * - minHeap: 存储较大的一半，堆顶是其中最小的
 *
 * 不变式：
 * 1. maxHeap.size() === minHeap.size() 或
 *    maxHeap.size() === minHeap.size() + 1
 * 2. maxHeap 中所有元素 ≤ minHeap 中所有元素
 *
 * @example
 * const median = new OnlineMedian();
 * median.add(1);
 * console.log(median.getMedian()); // 1
 * median.add(2);
 * console.log(median.getMedian()); // 1.5
 * median.add(3);
 * console.log(median.getMedian()); // 2
 */
export class OnlineMedian {
  private maxHeap: MaxHeap<number>; // 较小的一半
  private minHeap: MinHeap<number>; // 较大的一半

  constructor() {
    this.maxHeap = new MaxHeap<number>();
    this.minHeap = new MinHeap<number>();
  }

  /**
   * 添加数值
   *
   * 时间复杂度: O(log n)
   */
  add(num: number): void {
    // 1. 先加入 maxHeap（较小的一半）
    if (this.maxHeap.isEmpty() || num <= this.maxHeap.peek()!) {
      this.maxHeap.push(num);
    } else {
      this.minHeap.push(num);
    }

    // 2. 平衡两个堆的大小
    this.balance();
  }

  /**
   * 获取中位数
   *
   * 时间复杂度: O(1)
   */
  getMedian(): number {
    if (this.maxHeap.isEmpty()) {
      throw new Error('No data');
    }

    if (this.maxHeap.size() > this.minHeap.size()) {
      return this.maxHeap.peek()!;
    }

    return (this.maxHeap.peek()! + this.minHeap.peek()!) / 2;
  }

  /**
   * 数据数量
   */
  get count(): number {
    return this.maxHeap.size() + this.minHeap.size();
  }

  /**
   * 平衡两个堆
   */
  private balance(): void {
    // maxHeap 最多比 minHeap 多 1 个元素
    if (this.maxHeap.size() > this.minHeap.size() + 1) {
      this.minHeap.push(this.maxHeap.pop()!);
    } else if (this.minHeap.size() > this.maxHeap.size()) {
      this.maxHeap.push(this.minHeap.pop()!);
    }
  }
}

// ============================================================================
// 带删除功能的在线中位数（进阶）
// ============================================================================

/**
 * 支持删除的在线中位数
 *
 * 使用惰性删除策略：
 * - 删除时只标记，不立即从堆中移除
 * - 获取堆顶时检查是否已删除
 */
export class OnlineMedianWithRemoval {
  private maxHeap: MaxHeap<number>;
  private minHeap: MinHeap<number>;
  private deleted: Map<number, number>; // 值 -> 删除计数
  private maxHeapSize: number;
  private minHeapSize: number;

  constructor() {
    this.maxHeap = new MaxHeap<number>();
    this.minHeap = new MinHeap<number>();
    this.deleted = new Map();
    this.maxHeapSize = 0;
    this.minHeapSize = 0;
  }

  /**
   * 添加数值
   */
  add(num: number): void {
    if (this.maxHeapSize === 0 || num <= this.getMaxHeapTop()) {
      this.maxHeap.push(num);
      this.maxHeapSize++;
    } else {
      this.minHeap.push(num);
      this.minHeapSize++;
    }
    this.balance();
  }

  /**
   * 删除数值（惰性删除）
   */
  remove(num: number): void {
    const count = this.deleted.get(num) ?? 0;
    this.deleted.set(num, count + 1);

    // 判断应该从哪个堆中删除
    if (num <= this.getMaxHeapTop()) {
      this.maxHeapSize--;
    } else {
      this.minHeapSize--;
    }

    this.balance();
  }

  /**
   * 获取中位数
   */
  getMedian(): number {
    if (this.maxHeapSize + this.minHeapSize === 0) {
      throw new Error('No data');
    }

    if (this.maxHeapSize > this.minHeapSize) {
      return this.getMaxHeapTop();
    }

    return (this.getMaxHeapTop() + this.getMinHeapTop()) / 2;
  }

  /**
   * 获取 maxHeap 的有效堆顶
   */
  private getMaxHeapTop(): number {
    while (true) {
      const top = this.maxHeap.peek()!;
      const deleteCount = this.deleted.get(top) ?? 0;
      if (deleteCount === 0) {
        return top;
      }
      this.maxHeap.pop();
      if (deleteCount === 1) {
        this.deleted.delete(top);
      } else {
        this.deleted.set(top, deleteCount - 1);
      }
    }
  }

  /**
   * 获取 minHeap 的有效堆顶
   */
  private getMinHeapTop(): number {
    while (true) {
      const top = this.minHeap.peek()!;
      const deleteCount = this.deleted.get(top) ?? 0;
      if (deleteCount === 0) {
        return top;
      }
      this.minHeap.pop();
      if (deleteCount === 1) {
        this.deleted.delete(top);
      } else {
        this.deleted.set(top, deleteCount - 1);
      }
    }
  }

  /**
   * 平衡两个堆
   */
  private balance(): void {
    if (this.maxHeapSize > this.minHeapSize + 1) {
      this.minHeap.push(this.getMaxHeapTop());
      this.maxHeap.pop();
      this.maxHeapSize--;
      this.minHeapSize++;
    } else if (this.minHeapSize > this.maxHeapSize) {
      this.maxHeap.push(this.getMinHeapTop());
      this.minHeap.pop();
      this.minHeapSize--;
      this.maxHeapSize++;
    }
  }
}

