/**
 * 📝 题目：二叉树的右视图
 * 🔗 链接：https://leetcode.cn/problems/binary-tree-right-side-view/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：树、深度优先搜索、广度优先搜索、二叉树
 *
 * 📋 题目描述：
 * 给定一个二叉树的 根节点 root，想象自己站在它的右侧，
 * 按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
 *
 * 示例：
 * 输入：[1,2,3,null,5,null,4]
 *
 *        1         ← 看到 1
 *       / \
 *      2   3       ← 看到 3
 *       \   \
 *        5   4     ← 看到 4
 *
 * 输出：[1,3,4]
 */

// ============================================================
// 💡 思路分析
// ============================================================
//
// 两种思路：
// 1. BFS 层序遍历：每层最后一个节点就是右视图
// 2. DFS 前序遍历：优先访问右子树，每层第一个访问到的就是答案

// ============================================================
// 节点定义
// ============================================================

class TreeNode {
  val: number;
  left: TreeNode | null;
  right: TreeNode | null;
  constructor(val: number = 0, left?: TreeNode | null, right?: TreeNode | null) {
    this.val = val;
    this.left = left === undefined ? null : left;
    this.right = right === undefined ? null : right;
  }
}

// ============================================================
// 解法一：BFS 层序遍历（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(w)

/**
 * 📊 BFS 执行过程图解：
 *
 *        1
 *       / \
 *      2   3
 *       \   \
 *        5   4
 *
 * 第 1 层: [1]     → 最后一个是 1
 * 第 2 层: [2, 3]  → 最后一个是 3
 * 第 3 层: [5, 4]  → 最后一个是 4
 *
 * 结果: [1, 3, 4]
 */
function rightSideView_bfs(root: TreeNode | null): number[] {
  if (!root) return [];

  const result: number[] = [];
  const queue: TreeNode[] = [root];

  while (queue.length > 0) {
    const levelSize = queue.length;

    for (let i = 0; i < levelSize; i++) {
      const node = queue.shift()!;

      // 每层的最后一个节点
      if (i === levelSize - 1) {
        result.push(node.val);
      }

      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
  }

  return result;
}

// ============================================================
// 解法二：DFS 深度优先（根→右→左）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(h)

/**
 * 📊 DFS 执行过程图解：
 *
 *        1
 *       / \
 *      2   3
 *       \   \
 *        5   4
 *
 * 优先访问右子树的 DFS：
 *
 * 访问 1 (depth=0): result.length=0, 记录 1 → [1]
 * 访问 3 (depth=1): result.length=1, 记录 3 → [1, 3]
 * 访问 4 (depth=2): result.length=2, 记录 4 → [1, 3, 4]
 * 访问 2 (depth=1): result.length=3 > 1, 跳过
 * 访问 5 (depth=2): result.length=3 > 2, 跳过
 *
 * 结果: [1, 3, 4]
 *
 * 关键：result.length 就是当前已经记录的层数
 *       如果 depth === result.length，说明这是该层第一个被访问的节点
 */
function rightSideView_dfs(root: TreeNode | null): number[] {
  const result: number[] = [];

  function dfs(node: TreeNode | null, depth: number): void {
    if (!node) return;

    // 每层第一个访问到的节点（因为优先访问右子树）
    if (depth === result.length) {
      result.push(node.val);
    }

    // 先右后左
    dfs(node.right, depth + 1);
    dfs(node.left, depth + 1);
  }

  dfs(root, 0);
  return result;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法  | 时间  | 空间  | 特点                        |
 * |------|-------|-------|----------------------------|
 * | BFS  | O(n)  | O(w)  | 直观，每层最后一个          |
 * | DFS  | O(n)  | O(h)  | 巧妙，优先右子树            |
 *
 * w = 最大宽度，完全二叉树约 n/2
 * h = 树高，平衡树约 log n
 *
 * 面试推荐 BFS，更直观易懂
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. DFS 方法的判断条件：
 *    - 是 depth === result.length
 *    - 不是 depth === result.length - 1
 *    - result.length 表示已记录的层数
 *
 * 2. DFS 遍历顺序：
 *    - 右视图：先右后左
 *    - 左视图：先左后右
 *
 * 3. BFS 的层末判断：
 *    - i === levelSize - 1
 *    - 不是 i === queue.length - 1
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 二叉树的左视图 → 先左后右，或每层第一个
 * - 二叉树的层序遍历 → BFS 模板题
 * - 锯齿形层序遍历 → 层序 + 奇偶层反转
 * - 找每层最大值 → 层序 + 求最大
 *
 * 共同模式：层序遍历的变体
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 侧边栏菜单折叠时显示最右边的图标
 * 2. 组织架构树右侧预览
 * 3. 文件树右侧显示最新修改时间
 * 4. 评论树右侧显示最新回复
 */

export { rightSideView_bfs, rightSideView_dfs, TreeNode };
export default rightSideView_bfs;

