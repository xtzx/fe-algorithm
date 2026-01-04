/**
 * 链表排序算法
 *
 * 链表的特点决定了适合的排序算法：
 * - 不支持随机访问 → 快排不适合
 * - 修改指针成本低 → 归并排序 merge 步骤很高效
 * - 找中点需要遍历 → 快慢指针 O(n)
 */

// ============================================================================
// 类型定义
// ============================================================================

/**
 * 链表节点
 */
export interface ListNode<T> {
  value: T;
  next: ListNode<T> | null;
}

/**
 * 比较函数类型
 */
export type Comparator<T> = (a: T, b: T) => number;

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 创建链表节点
 */
export function createNode<T>(value: T, next: ListNode<T> | null = null): ListNode<T> {
  return { value, next };
}

/**
 * 从数组创建链表
 */
export function fromArray<T>(arr: T[]): ListNode<T> | null {
  if (arr.length === 0) return null;

  const head = createNode(arr[0]);
  let curr = head;

  for (let i = 1; i < arr.length; i++) {
    curr.next = createNode(arr[i]);
    curr = curr.next;
  }

  return head;
}

/**
 * 链表转数组
 */
export function toArray<T>(head: ListNode<T> | null): T[] {
  const result: T[] = [];
  let curr = head;

  while (curr) {
    result.push(curr.value);
    curr = curr.next;
  }

  return result;
}

/**
 * 获取链表长度
 */
export function getLength<T>(head: ListNode<T> | null): number {
  let length = 0;
  let curr = head;

  while (curr) {
    length++;
    curr = curr.next;
  }

  return length;
}

/**
 * 快慢指针找中点（返回前半部分的最后一个节点）
 */
function getMiddle<T>(head: ListNode<T>): ListNode<T> {
  let slow: ListNode<T> = head;
  let fast: ListNode<T> | null = head.next;

  while (fast && fast.next) {
    slow = slow.next!;
    fast = fast.next.next;
  }

  return slow;
}

/**
 * 合并两个有序链表（稳定）
 */
function merge<T>(
  l1: ListNode<T> | null,
  l2: ListNode<T> | null,
  cmp: Comparator<T>
): ListNode<T> | null {
  // 使用哨兵节点简化边界处理
  const dummy = createNode(null as unknown as T);
  let curr = dummy;

  while (l1 && l2) {
    // <= 0 保证稳定性（相等时取左边的）
    if (cmp(l1.value, l2.value) <= 0) {
      curr.next = l1;
      l1 = l1.next;
    } else {
      curr.next = l2;
      l2 = l2.next;
    }
    curr = curr.next;
  }

  // 连接剩余部分
  curr.next = l1 || l2;

  return dummy.next;
}

// ============================================================================
// 归并排序（推荐用于链表）
// ============================================================================

/**
 * 链表归并排序（自顶向下，递归版本）
 *
 * 时间复杂度：O(n log n)
 * 空间复杂度：O(log n) 递归栈空间
 * 稳定性：稳定
 *
 * @example
 * const list = fromArray([3, 1, 4, 1, 5, 9, 2, 6]);
 * const sorted = mergeSortLinkedList(list, (a, b) => a - b);
 * console.log(toArray(sorted)); // [1, 1, 2, 3, 4, 5, 6, 9]
 */
export function mergeSortLinkedList<T>(
  head: ListNode<T> | null,
  cmp: Comparator<T>
): ListNode<T> | null {
  // 基准情况：空链表或单节点
  if (!head || !head.next) {
    return head;
  }

  // 1. 找中点并断开
  const middle = getMiddle(head);
  const rightHead = middle.next;
  middle.next = null;

  // 2. 递归排序左右两半
  const left = mergeSortLinkedList(head, cmp);
  const right = mergeSortLinkedList(rightHead, cmp);

  // 3. 合并
  return merge(left, right, cmp);
}

/**
 * 链表归并排序（自底向上，迭代版本）
 *
 * 时间复杂度：O(n log n)
 * 空间复杂度：O(1) 真正的常数空间
 * 稳定性：稳定
 *
 * 适用场景：对空间要求严格时使用
 */
export function mergeSortLinkedListIterative<T>(
  head: ListNode<T> | null,
  cmp: Comparator<T>
): ListNode<T> | null {
  if (!head || !head.next) {
    return head;
  }

  const length = getLength(head);
  const dummy = createNode(null as unknown as T);
  dummy.next = head;

  // 子链表大小从 1 开始，每次翻倍
  for (let size = 1; size < length; size *= 2) {
    let prev = dummy;
    let curr: ListNode<T> | null = dummy.next;

    while (curr) {
      // 获取左半部分（size 个节点）
      const left = curr;
      const leftTail = split(left, size);

      // 获取右半部分（size 个节点）
      const right = leftTail ? leftTail.next : null;
      if (leftTail) leftTail.next = null;

      const rightTail = right ? split(right, size) : null;
      const nextStart = rightTail ? rightTail.next : null;
      if (rightTail) rightTail.next = null;

      // 合并左右部分
      const [mergedHead, mergedTail] = mergeWithTail(left, right, cmp);

      // 连接到结果链表
      prev.next = mergedHead;
      if (mergedTail) {
        prev = mergedTail;
      }

      curr = nextStart;
    }
  }

  return dummy.next;
}

/**
 * 分割链表，返回第 n 个节点（从 1 开始）
 */
function split<T>(head: ListNode<T> | null, n: number): ListNode<T> | null {
  let curr = head;
  for (let i = 1; i < n && curr; i++) {
    curr = curr.next;
  }
  return curr;
}

/**
 * 合并两个链表，同时返回尾节点
 */
function mergeWithTail<T>(
  l1: ListNode<T> | null,
  l2: ListNode<T> | null,
  cmp: Comparator<T>
): [ListNode<T> | null, ListNode<T> | null] {
  const dummy = createNode(null as unknown as T);
  let curr = dummy;

  while (l1 && l2) {
    if (cmp(l1.value, l2.value) <= 0) {
      curr.next = l1;
      l1 = l1.next;
    } else {
      curr.next = l2;
      l2 = l2.next;
    }
    curr = curr.next;
  }

  curr.next = l1 || l2;

  // 找到尾节点
  while (curr.next) {
    curr = curr.next;
  }

  return [dummy.next, curr];
}

// ============================================================================
// 插入排序（适合小链表或近乎有序的数据）
// ============================================================================

/**
 * 链表插入排序
 *
 * 时间复杂度：O(n²) 最坏，O(n) 近乎有序时
 * 空间复杂度：O(1)
 * 稳定性：稳定
 *
 * 适用场景：
 * - 小链表（n < 50）
 * - 数据近乎有序
 *
 * @example
 * const list = fromArray([3, 1, 2]);
 * const sorted = insertionSortLinkedList(list, (a, b) => a - b);
 * console.log(toArray(sorted)); // [1, 2, 3]
 */
export function insertionSortLinkedList<T>(
  head: ListNode<T> | null,
  cmp: Comparator<T>
): ListNode<T> | null {
  if (!head || !head.next) {
    return head;
  }

  // 使用哨兵节点
  const dummy = createNode(null as unknown as T);
  let curr: ListNode<T> | null = head;

  while (curr) {
    const next = curr.next;

    // 找到插入位置
    let prev = dummy;
    while (prev.next && cmp(prev.next.value, curr.value) < 0) {
      prev = prev.next;
    }

    // 插入
    curr.next = prev.next;
    prev.next = curr;

    curr = next;
  }

  return dummy.next;
}

// ============================================================================
// 混合排序（模拟 TimSort 思想）
// ============================================================================

const INSERTION_THRESHOLD = 32;

/**
 * 混合排序：小链表用插入排序，大链表用归并排序
 *
 * 时间复杂度：O(n log n)
 * 空间复杂度：O(log n)
 * 稳定性：稳定
 */
export function hybridSortLinkedList<T>(
  head: ListNode<T> | null,
  cmp: Comparator<T>
): ListNode<T> | null {
  const length = getLength(head);

  if (length <= INSERTION_THRESHOLD) {
    return insertionSortLinkedList(head, cmp);
  }

  return mergeSortLinkedList(head, cmp);
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '链表排序',
  algorithms: {
    mergeSort: {
      name: '归并排序（递归）',
      timeComplexity: 'O(n log n)',
      spaceComplexity: 'O(log n)',
      stable: true,
      推荐场景: ['通用链表排序', '需要稳定性'],
    },
    mergeSortIterative: {
      name: '归并排序（迭代）',
      timeComplexity: 'O(n log n)',
      spaceComplexity: 'O(1)',
      stable: true,
      推荐场景: ['空间要求严格', 'LeetCode 148'],
    },
    insertionSort: {
      name: '插入排序',
      timeComplexity: 'O(n²) / O(n)',
      spaceComplexity: 'O(1)',
      stable: true,
      推荐场景: ['小链表', '近乎有序数据'],
    },
  },
  relatedProblems: [
    '148. 排序链表',
    '147. 对链表进行插入排序',
    '21. 合并两个有序链表',
    '23. 合并K个升序链表',
  ],
};

// ============================================================================
// 测试（纯手写断言）
// ============================================================================

if (typeof process !== 'undefined' && process.argv[1]?.includes('链表排序')) {
  console.log('🧪 链表排序测试\n');

  const numCmp = (a: number, b: number) => a - b;

  // 测试用例
  const testCases = [
    { input: [], desc: '空链表' },
    { input: [1], desc: '单元素' },
    { input: [1, 2, 3], desc: '已排序' },
    { input: [3, 2, 1], desc: '逆序' },
    { input: [3, 1, 4, 1, 5, 9, 2, 6], desc: '随机' },
    { input: [1, 1, 1, 1], desc: '全重复' },
  ];

  let passed = 0;
  let failed = 0;

  for (const { input, desc } of testCases) {
    const expected = [...input].sort((a, b) => a - b);

    // 测试归并排序（递归）
    const list1 = fromArray(input);
    const sorted1 = mergeSortLinkedList(list1, numCmp);
    const result1 = toArray(sorted1);

    if (JSON.stringify(result1) === JSON.stringify(expected)) {
      console.log(`✅ 归并排序（递归）- ${desc}`);
      passed++;
    } else {
      console.log(`❌ 归并排序（递归）- ${desc}: 期望 ${expected}，得到 ${result1}`);
      failed++;
    }

    // 测试归并排序（迭代）
    const list2 = fromArray(input);
    const sorted2 = mergeSortLinkedListIterative(list2, numCmp);
    const result2 = toArray(sorted2);

    if (JSON.stringify(result2) === JSON.stringify(expected)) {
      console.log(`✅ 归并排序（迭代）- ${desc}`);
      passed++;
    } else {
      console.log(`❌ 归并排序（迭代）- ${desc}: 期望 ${expected}，得到 ${result2}`);
      failed++;
    }

    // 测试插入排序
    const list3 = fromArray(input);
    const sorted3 = insertionSortLinkedList(list3, numCmp);
    const result3 = toArray(sorted3);

    if (JSON.stringify(result3) === JSON.stringify(expected)) {
      console.log(`✅ 插入排序 - ${desc}`);
      passed++;
    } else {
      console.log(`❌ 插入排序 - ${desc}: 期望 ${expected}，得到 ${result3}`);
      failed++;
    }
  }

  console.log(`\n📊 测试结果: ${passed} 通过, ${failed} 失败`);
}

