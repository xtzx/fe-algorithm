/**
 * 📝 题目：反转链表 II
 * 🔗 链接：https://leetcode.cn/problems/reverse-linked-list-ii/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：链表
 *
 * 📋 题目描述：
 * 给你单链表的头指针 head 和两个整数 left 和 right，
 * 其中 left <= right。请你反转从位置 left 到位置 right 的链表节点，
 * 返回反转后的链表。
 *
 * 示例：
 * 输入：head = [1,2,3,4,5], left = 2, right = 4
 * 输出：[1,4,3,2,5]
 *
 * 输入：head = [5], left = 1, right = 1
 * 输出：[5]
 */

// 链表节点定义
class ListNode {
  val: number;
  next: ListNode | null;
  constructor(val?: number, next?: ListNode | null) {
    this.val = val === undefined ? 0 : val;
    this.next = next === undefined ? null : next;
  }
}

// ============================================================
// 💡 思路分析
// ============================================================
//
// 这道题是「反转链表」的进阶版：只反转一部分
//
// 关键点：
// 1. 找到 left 的前一个节点（用于连接）
// 2. 反转 [left, right] 部分
// 3. 连接前后部分
//
// 两种方法：
// 1. 头插法：每次把当前节点插到反转部分的最前面
// 2. 先截取再反转：截出来反转后再接回去

// ============================================================
// 解法一：头插法（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(1)

/**
 * 📊 头插法执行过程图解：
 *
 * 输入: 1 -> 2 -> 3 -> 4 -> 5, left = 2, right = 4
 *
 * 使用 dummy 节点：
 * dummy -> 1 -> 2 -> 3 -> 4 -> 5
 *
 * 找到 left 前一个节点 prev：
 * dummy -> 1 -> 2 -> 3 -> 4 -> 5
 *          ↑    ↑
 *         prev curr
 *
 * 第一次操作（把 3 插到 2 前面）：
 * dummy -> 1 -> 3 -> 2 -> 4 -> 5
 *          ↑         ↑    ↑
 *         prev     curr  next
 *
 * 第二次操作（把 4 插到 3 前面）：
 * dummy -> 1 -> 4 -> 3 -> 2 -> 5
 *          ↑              ↑    ↑
 *         prev          curr  next
 *
 * 结果: 1 -> 4 -> 3 -> 2 -> 5
 */
function reverseBetween_v1(
  head: ListNode | null,
  left: number,
  right: number
): ListNode | null {
  if (!head || left === right) return head;

  // 虚拟头节点，处理 left = 1 的情况
  const dummy = new ListNode(0);
  dummy.next = head;

  // 找到 left 的前一个节点
  let prev: ListNode = dummy;
  for (let i = 0; i < left - 1; i++) {
    prev = prev.next!;
  }

  // curr 是反转部分的第一个节点（最终会变成反转部分的最后一个）
  const curr = prev.next!;

  // 头插法：每次把 curr 的下一个节点插到 prev 后面
  for (let i = 0; i < right - left; i++) {
    const next = curr.next!;
    curr.next = next.next;
    next.next = prev.next;
    prev.next = next;
  }

  return dummy.next;
}

// ============================================================
// 解法二：截取后反转再接回
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(1)

/**
 * 📊 截取反转思路图解：
 *
 * 原链表: 1 -> 2 -> 3 -> 4 -> 5
 *
 * Step 1: 找到四个关键节点
 * - prev: left 的前一个节点 (1)
 * - leftNode: left 节点 (2)
 * - rightNode: right 节点 (4)
 * - succ: right 的下一个节点 (5)
 *
 * Step 2: 截取 [left, right]
 * 1 -> 2 -> 3 -> 4    5
 *
 * Step 3: 反转截取的部分
 * 1 -> 4 -> 3 -> 2    5
 *
 * Step 4: 接回去
 * 1 -> 4 -> 3 -> 2 -> 5
 */
function reverseBetween_v2(
  head: ListNode | null,
  left: number,
  right: number
): ListNode | null {
  if (!head || left === right) return head;

  const dummy = new ListNode(0);
  dummy.next = head;

  // 找到 left 的前一个节点
  let prev: ListNode = dummy;
  for (let i = 0; i < left - 1; i++) {
    prev = prev.next!;
  }

  // 找到 right 节点
  let rightNode: ListNode = prev;
  for (let i = 0; i < right - left + 1; i++) {
    rightNode = rightNode.next!;
  }

  // 截取
  const leftNode = prev.next!;
  const succ = rightNode.next;

  // 断开连接
  prev.next = null;
  rightNode.next = null;

  // 反转
  reverseList(leftNode);

  // 接回去
  prev.next = rightNode; // rightNode 现在是头
  leftNode.next = succ; // leftNode 现在是尾

  return dummy.next;
}

// 辅助函数：反转链表
function reverseList(head: ListNode | null): ListNode | null {
  let prev: ListNode | null = null;
  let curr: ListNode | null = head;

  while (curr) {
    const next = curr.next;
    curr.next = prev;
    prev = curr;
    curr = next;
  }

  return prev;
}

// ============================================================
// 解法三：递归
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n) 递归栈

function reverseBetween_v3(
  head: ListNode | null,
  left: number,
  right: number
): ListNode | null {
  // left = 1 时，就是反转前 right 个节点
  if (left === 1) {
    return reverseN(head, right);
  }

  // left > 1 时，递归处理
  head!.next = reverseBetween_v3(head!.next, left - 1, right - 1);
  return head;
}

// 后驱节点
let successor: ListNode | null = null;

// 反转前 n 个节点
function reverseN(head: ListNode | null, n: number): ListNode | null {
  if (n === 1) {
    successor = head!.next;
    return head;
  }

  const newHead = reverseN(head!.next, n - 1);
  head!.next!.next = head;
  head!.next = successor;

  return newHead;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法       | 时间  | 空间  | 特点                        |
 * |-----------|-------|-------|----------------------------|
 * | 头插法    | O(n)  | O(1)  | 推荐，一次遍历              |
 * | 截取反转  | O(n)  | O(1)  | 思路清晰，步骤多            |
 * | 递归      | O(n)  | O(n)  | 优雅但空间开销大            |
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 使用 dummy 节点：
 *    - left = 1 时头节点会变化
 *    - 用 dummy 统一处理
 *
 * 2. 头插法的核心操作：
 *    - curr 不动，每次把 curr.next 移到 prev.next
 *    - 需要三步：
 *      a. curr.next = next.next（跳过 next）
 *      b. next.next = prev.next（next 指向当前头）
 *      c. prev.next = next（更新头）
 *
 * 3. 循环次数：
 *    - 需要操作 right - left 次
 *    - 不是 right - left + 1
 *
 * 4. 截取法要断开连接：
 *    - prev.next = null
 *    - rightNode.next = null
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 反转链表 → 基础版本
 * - K 个一组翻转链表 → 分组反转
 * - 回文链表 → 反转后半部分
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 拖拽排序：拖动元素到新位置
 * 2. 列表操作：反转部分列表顺序
 * 3. 撤销重做：反转操作历史
 * 4. 动画序列：反转部分动画顺序
 */

export { ListNode, reverseBetween_v1, reverseBetween_v2, reverseBetween_v3 };
export default reverseBetween_v1;

