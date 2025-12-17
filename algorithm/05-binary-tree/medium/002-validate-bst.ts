/**
 * 📝 题目：验证二叉搜索树
 * 🔗 链接：https://leetcode.cn/problems/validate-binary-search-tree/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：树、深度优先搜索、二叉搜索树、二叉树
 *
 * 📋 题目描述：
 * 给你一个二叉树的根节点 root，判断其是否是一个有效的二叉搜索树。
 *
 * 有效 二叉搜索树定义如下：
 * - 节点的左子树只包含 小于 当前节点的数。
 * - 节点的右子树只包含 大于 当前节点的数。
 * - 所有左子树和右子树自身必须也是二叉搜索树。
 *
 * 示例：
 *       5
 *      / \
 *     1   4
 *        / \
 *       3   6
 *
 * 输入：root = [5,1,4,null,null,3,6]
 * 输出：false
 * 解释：根节点的值是 5，但是右子节点的值是 4（4 < 5）
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
// 错误思路：只比较节点和它的直接子节点
//   5
//  / \
// 1   6
//    / \
//   3   7
// 这棵树不是 BST（3 < 5），但每个节点都比直接左子节点大
//
// 正确思路：
// 1. 递归 + 区间限制：每个节点需要在一个有效区间内
// 2. 中序遍历：BST 的中序遍历是严格递增的

// ============================================================
// 解法一：递归 + 区间限制（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(h)

/**
 * 📊 区间限制图解：
 *
 *       5 (min=-∞, max=+∞)
 *      / \
 *     1   6
 *    (min=-∞, max=5)  (min=5, max=+∞)
 *
 * 每个节点的值必须在 (min, max) 区间内
 * 进入左子树：更新 max = node.val
 * 进入右子树：更新 min = node.val
 */
function isValidBST_v1(root: TreeNode | null): boolean {
  return validate(root, -Infinity, Infinity);
}

function validate(
  node: TreeNode | null,
  min: number,
  max: number
): boolean {
  if (!node) return true;

  // 当前节点必须在 (min, max) 区间内
  if (node.val <= min || node.val >= max) {
    return false;
  }

  // 递归检查左右子树
  return (
    validate(node.left, min, node.val) &&
    validate(node.right, node.val, max)
  );
}

// ============================================================
// 解法二：中序遍历
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(h)

/**
 * 📊 中序遍历图解：
 *
 * BST 的中序遍历结果是严格递增的
 *
 *       5
 *      / \
 *     3   7
 *    / \
 *   2   4
 *
 * 中序：2 → 3 → 4 → 5 → 7（递增）
 *
 * 验证：每个节点必须大于前一个访问的节点
 */
function isValidBST_v2(root: TreeNode | null): boolean {
  let prev = -Infinity;

  function inorder(node: TreeNode | null): boolean {
    if (!node) return true;

    // 先遍历左子树
    if (!inorder(node.left)) return false;

    // 检查当前节点是否大于前一个节点
    if (node.val <= prev) return false;
    prev = node.val;

    // 再遍历右子树
    return inorder(node.right);
  }

  return inorder(root);
}

// ============================================================
// 解法三：中序遍历（迭代）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(h)
function isValidBST_v3(root: TreeNode | null): boolean {
  const stack: TreeNode[] = [];
  let prev = -Infinity;
  let curr = root;

  while (curr || stack.length > 0) {
    // 一直走到最左
    while (curr) {
      stack.push(curr);
      curr = curr.left;
    }

    curr = stack.pop()!;

    // 检查是否递增
    if (curr.val <= prev) return false;
    prev = curr.val;

    curr = curr.right;
  }

  return true;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法         | 时间  | 空间  | 特点                     |
 * |-------------|-------|-------|-------------------------|
 * | 区间限制     | O(n)  | O(h)  | 推荐，思路清晰            |
 * | 中序递归     | O(n)  | O(h)  | 利用 BST 性质            |
 * | 中序迭代     | O(n)  | O(h)  | 避免递归栈溢出            |
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 区间是开区间：
 *    - node.val <= min 或 node.val >= max 都不行
 *    - 不能有相等的情况
 *
 * 2. 不能只比较直接子节点：
 *    - 需要保证整个子树都满足条件
 *
 * 3. 初始区间：
 *    - min = -Infinity, max = Infinity
 *    - 或者用 null 表示无限制
 *
 * 4. 中序遍历的 prev：
 *    - 初始为 -Infinity
 *    - 要用严格小于，不能相等
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - BST 中第 K 小的元素 → 中序遍历
 * - 二叉搜索树的最近公共祖先 → 利用 BST 性质
 * - 将有序数组转换为 BST → 构造题
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 数据验证：验证数据结构是否符合规则
 * 2. 搜索优化：利用 BST 进行快速查找
 */

export { TreeNode, isValidBST_v1, isValidBST_v2, isValidBST_v3 };
export default isValidBST_v1;

