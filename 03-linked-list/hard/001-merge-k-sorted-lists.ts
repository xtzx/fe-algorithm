/**
 * 📝 题目：合并 K 个升序链表
 * 🔗 链接：https://leetcode.cn/problems/merge-k-sorted-lists/
 * 🏷️ 难度：Hard
 * 🏷️ 标签：链表、分治、堆（优先队列）、归并排序
 *
 * 📋 题目描述：
 * 给你一个链表数组，每个链表都已经按升序排列。
 * 请你将所有链表合并到一个升序链表中，返回合并后的链表。
 *
 * 示例：
 * 输入：lists = [[1,4,5],[1,3,4],[2,6]]
 * 输出：[1,1,2,3,4,4,5,6]
 *
 * 解释：
 * 链表数组如下：
 * [
 *   1->4->5,
 *   1->3->4,
 *   2->6
 * ]
 * 将它们合并到一个有序链表中得到。
 * 1->1->2->3->4->4->5->6
 */

class ListNode {
  val: number;
  next: ListNode | null;
  constructor(val?: number, next?: ListNode | null) {
    this.val = val === undefined ? 0 : val;
    this.next = next === undefined ? null : next;
  }
}

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 从"合并两个有序链表"扩展到"合并 K 个"
//
// 1. 暴力法：逐一合并
//    - 依次将链表两两合并
//    - 时间 O(kN)，k 是链表数量，N 是总节点数
//
// 2. 分治法（推荐）：
//    - 类似归并排序的分治思想
//    - 两两合并，层层向上
//    - 时间 O(N log k)
//
// 3. 优先队列（最小堆）：
//    - 用堆维护 k 个链表的当前头节点
//    - 每次取最小的，然后把该链表的下一个节点入堆
//    - 时间 O(N log k)

// ============================================================
// 解法一：分治法（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(N log k) | 空间复杂度：O(log k) 递归栈

/**
 * 📊 分治过程图解：
 *
 * 输入：[l1, l2, l3, l4, l5, l6]
 *
 * 第一层分治：
 *   [l1, l2, l3] | [l4, l5, l6]
 *
 * 第二层分治：
 *   [l1, l2] | [l3] | [l4, l5] | [l6]
 *
 * 第三层分治：
 *   [l1] | [l2] | [l3] | [l4] | [l5] | [l6]
 *
 * 合并回溯：
 *   merge(l1, l2) -> l12
 *   merge(l12, l3) -> l123
 *   merge(l4, l5) -> l45
 *   merge(l45, l6) -> l456
 *   merge(l123, l456) -> result
 *
 * 🔄 流程图 (Mermaid):
 * ```mermaid
 * flowchart TD
 *     A[lists 数组] --> B{length <= 1?}
 *     B -->|Yes| C[返回 lists 0 或 null]
 *     B -->|No| D[mid = length / 2]
 *     D --> E[递归处理左半: 0 to mid]
 *     D --> F[递归处理右半: mid to end]
 *     E --> G[合并左右结果]
 *     F --> G
 *     G --> H[返回合并结果]
 * ```
 */
function mergeKLists_v1(lists: Array<ListNode | null>): ListNode | null {
  if (lists.length === 0) return null;
  if (lists.length === 1) return lists[0];

  return divide(lists, 0, lists.length - 1);
}

function divide(
  lists: Array<ListNode | null>,
  left: number,
  right: number
): ListNode | null {
  if (left === right) {
    return lists[left];
  }

  const mid = Math.floor((left + right) / 2);
  const l1 = divide(lists, left, mid);
  const l2 = divide(lists, mid + 1, right);

  return mergeTwoLists(l1, l2);
}

function mergeTwoLists(
  l1: ListNode | null,
  l2: ListNode | null
): ListNode | null {
  const dummy = new ListNode(0);
  let curr = dummy;

  while (l1 && l2) {
    if (l1.val <= l2.val) {
      curr.next = l1;
      l1 = l1.next;
    } else {
      curr.next = l2;
      l2 = l2.next;
    }
    curr = curr.next;
  }

  curr.next = l1 || l2;
  return dummy.next;
}

// ============================================================
// 解法二：优先队列（最小堆）
// ============================================================
// ⏱️ 时间复杂度：O(N log k) | 空间复杂度：O(k)

/**
 * 📊 优先队列过程图解：
 *
 * lists = [[1,4,5], [1,3,4], [2,6]]
 *
 * 初始堆（按值排序）：
 *   [1(l1), 1(l2), 2(l3)]  // 三个链表的头节点
 *
 * Step 1: 弹出 1(l1)，加入结果，把 l1 的下一个节点 4 入堆
 *   结果: 1
 *   堆: [1(l2), 2(l3), 4(l1)]
 *
 * Step 2: 弹出 1(l2)，加入结果，把 l2 的下一个节点 3 入堆
 *   结果: 1 -> 1
 *   堆: [2(l3), 4(l1), 3(l2)]
 *
 * Step 3: 弹出 2(l3)，加入结果，把 l3 的下一个节点 6 入堆
 *   结果: 1 -> 1 -> 2
 *   堆: [3(l2), 4(l1), 6(l3)]
 *
 * ... 继续直到堆为空
 */

// JavaScript 没有内置堆，需要自己实现
class MinHeap {
  private heap: ListNode[] = [];

  size(): number {
    return this.heap.length;
  }

  push(node: ListNode): void {
    this.heap.push(node);
    this.bubbleUp(this.heap.length - 1);
  }

  pop(): ListNode | undefined {
    if (this.heap.length === 0) return undefined;
    if (this.heap.length === 1) return this.heap.pop();

    const result = this.heap[0];
    this.heap[0] = this.heap.pop()!;
    this.bubbleDown(0);
    return result;
  }

  private bubbleUp(index: number): void {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      if (this.heap[parentIndex].val <= this.heap[index].val) break;
      [this.heap[parentIndex], this.heap[index]] = [
        this.heap[index],
        this.heap[parentIndex],
      ];
      index = parentIndex;
    }
  }

  private bubbleDown(index: number): void {
    while (true) {
      let smallest = index;
      const left = 2 * index + 1;
      const right = 2 * index + 2;

      if (left < this.heap.length && this.heap[left].val < this.heap[smallest].val) {
        smallest = left;
      }
      if (right < this.heap.length && this.heap[right].val < this.heap[smallest].val) {
        smallest = right;
      }

      if (smallest === index) break;

      [this.heap[smallest], this.heap[index]] = [
        this.heap[index],
        this.heap[smallest],
      ];
      index = smallest;
    }
  }
}

function mergeKLists_v2(lists: Array<ListNode | null>): ListNode | null {
  const heap = new MinHeap();

  // 把所有链表的头节点入堆
  for (const head of lists) {
    if (head) {
      heap.push(head);
    }
  }

  const dummy = new ListNode(0);
  let curr = dummy;

  while (heap.size() > 0) {
    const node = heap.pop()!;
    curr.next = node;
    curr = curr.next;

    // 把该链表的下一个节点入堆
    if (node.next) {
      heap.push(node.next);
    }
  }

  return dummy.next;
}

// ============================================================
// 解法三：逐一合并（暴力法，不推荐）
// ============================================================
// ⏱️ 时间复杂度：O(kN) | 空间复杂度：O(1)
function mergeKLists_v3(lists: Array<ListNode | null>): ListNode | null {
  if (lists.length === 0) return null;

  let result = lists[0];
  for (let i = 1; i < lists.length; i++) {
    result = mergeTwoLists(result, lists[i]);
  }

  return result;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法       | 时间        | 空间       | 特点                  |
 * |-----------|-------------|------------|----------------------|
 * | 分治      | O(N log k)  | O(log k)   | 推荐，代码简洁         |
 * | 优先队列   | O(N log k)  | O(k)       | 适合流式数据           |
 * | 逐一合并   | O(kN)       | O(1)       | 简单但慢              |
 *
 * k = 链表数量，N = 总节点数
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 边界情况：
 *    - lists 为空数组
 *    - lists 中有 null 元素
 *
 * 2. 分治的递归终止条件：
 *    - left === right 时返回 lists[left]
 *
 * 3. 优先队列的实现：
 *    - JavaScript 没有内置堆，需要自己实现
 *    - 或使用第三方库
 *
 * 4. 空间复杂度分析：
 *    - 分治：递归栈深度 O(log k)
 *    - 优先队列：堆大小 O(k)
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 合并两个有序链表 → 基础问题
 * - 合并区间 → 类似的合并思想
 * - 丑数 II → 多路归并
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 多数据源合并：合并多个 API 返回的有序数据
 * 2. 日志归并：合并多个服务的时间有序日志
 * 3. 搜索结果合并：合并多个搜索引擎的结果
 * 4. 实时数据流：合并多个有序事件流
 */

export { ListNode, MinHeap, mergeKLists_v1, mergeKLists_v2, mergeKLists_v3 };
export default mergeKLists_v1;

