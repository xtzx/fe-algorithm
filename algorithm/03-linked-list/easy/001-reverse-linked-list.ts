/**
 * 📝 题目：反转链表
 * 🔗 链接：https://leetcode.cn/problems/reverse-linked-list/
 * 🏷️ 难度：Easy
 * 🏷️ 标签：递归、链表
 *
 * 📋 题目描述：
 * 给你单链表的头节点 head，请你反转链表，并返回反转后的链表。
 *
 * 示例：
 * 输入：head = [1,2,3,4,5]
 * 输出：[5,4,3,2,1]
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
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 链表反转是最基础、最重要的链表操作！
//
// 核心：改变每个节点的 next 指针方向
//
// 1. 迭代法：
//    - 维护三个指针：prev, curr, next
//    - 遍历链表，逐个反转指针
//
// 2. 递归法：
//    - 先递归到链表末尾
//    - 回溯时反转指针

// ============================================================
// 解法一：迭代法（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(1)

/**
 * 📊 执行过程图解：
 *
 * 初始状态：
 *   null    1 -> 2 -> 3 -> 4 -> 5 -> null
 *    ↑      ↑
 *   prev   curr
 *
 * Step 1: 保存 next，反转指针
 *   null <- 1    2 -> 3 -> 4 -> 5 -> null
 *           ↑    ↑
 *          prev curr
 *
 * Step 2:
 *   null <- 1 <- 2    3 -> 4 -> 5 -> null
 *                ↑    ↑
 *               prev curr
 *
 * Step 3:
 *   null <- 1 <- 2 <- 3    4 -> 5 -> null
 *                     ↑    ↑
 *                    prev curr
 *
 * Step 4:
 *   null <- 1 <- 2 <- 3 <- 4    5 -> null
 *                          ↑    ↑
 *                         prev curr
 *
 * Step 5:
 *   null <- 1 <- 2 <- 3 <- 4 <- 5    null
 *                               ↑     ↑
 *                              prev  curr
 *
 * 结束：curr == null，返回 prev
 *
 * 🔄 流程图 (Mermaid):
 * ```mermaid
 * flowchart TD
 *     A[prev=null, curr=head] --> B{curr != null?}
 *     B -->|Yes| C[next = curr.next]
 *     C --> D[curr.next = prev]
 *     D --> E[prev = curr]
 *     E --> F[curr = next]
 *     F --> B
 *     B -->|No| G[返回 prev]
 * ```
 */
function reverseList_v1(head: ListNode | null): ListNode | null {
  let prev: ListNode | null = null;
  let curr: ListNode | null = head;

  while (curr) {
    const next = curr.next; // 1. 保存下一个节点
    curr.next = prev; // 2. 反转指针
    prev = curr; // 3. 移动 prev
    curr = next; // 4. 移动 curr
  }

  return prev; // prev 指向新的头节点
}

// ============================================================
// 解法二：递归法
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n) 递归栈

/**
 * 📊 递归过程图解：
 *
 * 递归到底部：
 *   reverseList(1) -> reverseList(2) -> reverseList(3) -> reverseList(4) -> reverseList(5)
 *                                                                              ↓
 *                                                                           返回 5
 *
 * 回溯过程（以 node=4 为例）：
 *
 *   4 -> 5 -> null
 *
 *   head.next.next = head  →  4 -> 5 -> 4
 *                             (5.next = 4)
 *
 *   head.next = null       →  null <- 4 <- 5
 *
 *   返回 5（新的头节点）
 */
function reverseList_v2(head: ListNode | null): ListNode | null {
  // 基本情况：空链表或只有一个节点
  if (!head || !head.next) {
    return head;
  }

  // 递归反转后面的部分
  const newHead = reverseList_v2(head.next);

  // 反转当前节点的指针
  head.next.next = head; // 让下一个节点指向自己
  head.next = null; // 断开原来的连接

  return newHead; // 返回新的头节点
}

// ============================================================
// 解法三：用栈（不推荐，仅供理解）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n)
function reverseList_v3(head: ListNode | null): ListNode | null {
  if (!head) return null;

  const stack: ListNode[] = [];

  // 把所有节点入栈
  let curr: ListNode | null = head;
  while (curr) {
    stack.push(curr);
    curr = curr.next;
  }

  // 出栈重建链表
  const newHead = stack.pop()!;
  curr = newHead;

  while (stack.length > 0) {
    curr.next = stack.pop()!;
    curr = curr.next;
  }
  curr.next = null;

  return newHead;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法     | 时间  | 空间  | 特点                     |
 * |---------|-------|-------|-------------------------|
 * | 迭代    | O(n)  | O(1)  | 推荐，空间最优            |
 * | 递归    | O(n)  | O(n)  | 代码简洁，理解递归        |
 * | 栈      | O(n)  | O(n)  | 思路直观，但不推荐        |
 *
 * 面试建议：
 * 1. 首选迭代法（效率最高）
 * 2. 能写出递归法（展示递归能力）
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 迭代法中指针操作的顺序：
 *    - 必须先保存 next，再反转
 *    - 否则 curr.next 被覆盖，链表断裂
 *
 * 2. 递归法的基本情况：
 *    - !head || !head.next
 *    - 空链表或单节点直接返回
 *
 * 3. 递归法中的断开连接：
 *    - head.next = null 很重要
 *    - 否则原来的头节点会形成环
 *
 * 4. 返回值：
 *    - 迭代返回 prev（新的头）
 *    - 递归返回 newHead（不变）
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 反转链表 II（反转部分链表）→ 迭代，记录边界
 * - 回文链表 → 反转后半部分比较
 * - K 个一组翻转链表 → 分段反转
 *
 * 链表反转是很多链表题的基础操作！
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 撤销功能：操作历史的反转
 * 2. 路由历史：浏览器前进后退
 * 3. 消息列表：按时间倒序展示
 * 4. 日志展示：最新的在最前面
 */

export { ListNode, reverseList_v1, reverseList_v2, reverseList_v3 };
export default reverseList_v1;

