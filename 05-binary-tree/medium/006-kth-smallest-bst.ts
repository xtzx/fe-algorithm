/**
 * 📝 题目：二叉搜索树中第 K 小的元素
 * 🔗 链接：https://leetcode.cn/problems/kth-smallest-element-in-a-bst/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：树、深度优先搜索、二叉搜索树、二叉树
 *
 * 📋 题目描述：
 * 给定一个二叉搜索树的根节点 root，和一个整数 k，
 * 请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。
 *
 * 示例：
 *       3
 *      / \
 *     1   4
 *      \
 *       2
 *
 * 输入：root = [3,1,4,null,2], k = 1
 * 输出：1
 *
 *          5
 *         / \
 *        3   6
 *       / \
 *      2   4
 *     /
 *    1
 *
 * 输入：root = [5,3,6,2,4,null,null,1], k = 3
 * 输出：3
 */

class TreeNode {
  val: number;
  left: TreeNode | null;
  right: TreeNode | null;
  constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
    this.val = val === undefined ? 0 : val;
    this.left = left === undefined ? null : left;
    this.right = right === undefined ? null : right;
  }
}

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// BST 的重要性质：中序遍历结果是有序的！
//
// 方法一：中序遍历，数到第 k 个
// 方法二：中序遍历，提前返回（优化）

// ============================================================
// 解法一：中序遍历（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(k) 平均，O(n) 最坏 | 空间复杂度：O(h)

/**
 * 📊 BST 中序遍历图解：
 *
 *       3
 *      / \
 *     1   4
 *      \
 *       2
 *
 * 中序遍历: 1 → 2 → 3 → 4（递增！）
 *
 * 第 1 小：1
 * 第 2 小：2
 * 第 3 小：3
 * 第 4 小：4
 */
function kthSmallest_v1(root: TreeNode | null, k: number): number {
  let count = 0;
  let result = 0;

  function inorder(node: TreeNode | null) {
    if (!node) return;

    // 遍历左子树
    inorder(node.left);

    // 访问当前节点（中序位置）
    count++;
    if (count === k) {
      result = node.val;
      return; // 找到后可以提前返回
    }

    // 遍历右子树
    inorder(node.right);
  }

  inorder(root);
  return result;
}

// ============================================================
// 解法二：迭代版中序遍历
// ============================================================
// ⏱️ 时间复杂度：O(k) | 空间复杂度：O(h)
function kthSmallest_v2(root: TreeNode | null, k: number): number {
  const stack: TreeNode[] = [];
  let curr = root;
  let count = 0;

  while (curr || stack.length > 0) {
    // 一直走到最左
    while (curr) {
      stack.push(curr);
      curr = curr.left;
    }

    // 访问节点
    curr = stack.pop()!;
    count++;

    if (count === k) {
      return curr.val;
    }

    // 转向右子树
    curr = curr.right;
  }

  return -1; // 不会到达这里
}

// ============================================================
// 解法三：转数组（不推荐，但简单）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n)
function kthSmallest_v3(root: TreeNode | null, k: number): number {
  const arr: number[] = [];

  function inorder(node: TreeNode | null) {
    if (!node) return;
    inorder(node.left);
    arr.push(node.val);
    inorder(node.right);
  }

  inorder(root);
  return arr[k - 1];
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法         | 时间  | 空间  | 特点                     |
 * |-------------|-------|-------|-------------------------|
 * | 递归中序     | O(k)* | O(h)  | 推荐                     |
 * | 迭代中序     | O(k)  | O(h)  | 可以真正提前停止          |
 * | 转数组       | O(n)  | O(n)  | 简单但不优               |
 *
 * * 递归版本理论上可以提前返回，但递归栈会继续回溯
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. k 从 1 开始计数：
 *    - 第 1 小是最小的，不是索引 0
 *
 * 2. BST 性质：
 *    - 中序遍历是递增的
 *    - 不是前序或后序！
 *
 * 3. 提前返回优化：
 *    - 找到后不需要继续遍历
 *    - 迭代版本可以直接 return
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 二叉搜索树的第 k 大元素 → 反向中序遍历
 * - 二叉搜索树中的众数 → 中序遍历找众数
 * - 二叉搜索树的范围和 → 中序遍历求范围内的和
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 排行榜：获取第 K 名
 * 2. 数据排序：BST 天然支持排序
 */

export { TreeNode, kthSmallest_v1, kthSmallest_v2, kthSmallest_v3 };
export default kthSmallest_v1;

