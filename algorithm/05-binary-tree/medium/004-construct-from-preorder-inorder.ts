/**
 * 📝 题目：从前序与中序遍历序列构造二叉树
 * 🔗 链接：https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：树、数组、哈希表、分治、二叉树
 *
 * 📋 题目描述：
 * 给定两个整数数组 preorder 和 inorder，
 * 其中 preorder 是二叉树的先序遍历，inorder 是同一棵树的中序遍历，
 * 请构造二叉树并返回其根节点。
 *
 * 示例：
 * preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
 *
 *        3
 *       / \
 *      9  20
 *        /  \
 *       15   7
 *
 * 输入：preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
 * 输出：[3,9,20,null,null,15,7]
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
// 关键观察：
// 1. 前序遍历的第一个元素是根节点
// 2. 在中序遍历中找到根节点，左边是左子树，右边是右子树
// 3. 根据中序中左子树的长度，可以在前序中分出左子树和右子树
//
// 递归构造：
// 1. 从前序取根节点
// 2. 在中序中找到根节点位置
// 3. 递归构造左子树和右子树

// ============================================================
// 解法：递归 + 哈希表优化
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n)

/**
 * 📊 构造过程图解：
 *
 * preorder = [3, 9, 20, 15, 7]
 * inorder  = [9, 3, 15, 20, 7]
 *
 * Step 1: 前序第一个 3 是根
 *         中序中 3 的位置是 1
 *         左子树：inorder[0:1] = [9]，preorder[1:2] = [9]
 *         右子树：inorder[2:5] = [15,20,7]，preorder[2:5] = [20,15,7]
 *
 *        3
 *       / \
 *      ?   ?
 *
 * Step 2: 构造左子树
 *         preorder = [9], inorder = [9]
 *         根是 9，无左右子树
 *
 *        3
 *       / \
 *      9   ?
 *
 * Step 3: 构造右子树
 *         preorder = [20,15,7], inorder = [15,20,7]
 *         根是 20
 *         左子树：[15]，右子树：[7]
 *
 *        3
 *       / \
 *      9  20
 *        /  \
 *       15   7
 */
function buildTree(preorder: number[], inorder: number[]): TreeNode | null {
  // 用哈希表存储中序遍历的值和索引，O(1) 查找
  const inorderMap = new Map<number, number>();
  inorder.forEach((val, index) => inorderMap.set(val, index));

  function build(
    preStart: number,
    preEnd: number,
    inStart: number,
    inEnd: number
  ): TreeNode | null {
    // Base Case: 区间为空
    if (preStart > preEnd) return null;

    // 前序遍历的第一个元素是根节点
    const rootVal = preorder[preStart];
    const root = new TreeNode(rootVal);

    // 在中序遍历中找到根节点的位置
    const rootIndex = inorderMap.get(rootVal)!;

    // 左子树的节点数量
    const leftSize = rootIndex - inStart;

    // 递归构造左子树
    // 前序：[preStart+1, preStart+leftSize]
    // 中序：[inStart, rootIndex-1]
    root.left = build(
      preStart + 1,
      preStart + leftSize,
      inStart,
      rootIndex - 1
    );

    // 递归构造右子树
    // 前序：[preStart+leftSize+1, preEnd]
    // 中序：[rootIndex+1, inEnd]
    root.right = build(
      preStart + leftSize + 1,
      preEnd,
      rootIndex + 1,
      inEnd
    );

    return root;
  }

  return build(0, preorder.length - 1, 0, inorder.length - 1);
}

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 区间划分：
 *    - 前序中左子树：[preStart+1, preStart+leftSize]
 *    - 前序中右子树：[preStart+leftSize+1, preEnd]
 *    - 中序中左子树：[inStart, rootIndex-1]
 *    - 中序中右子树：[rootIndex+1, inEnd]
 *
 * 2. leftSize 的计算：
 *    - leftSize = rootIndex - inStart
 *    - 不是 rootIndex
 *
 * 3. 使用哈希表优化：
 *    - 不用哈希表，每次在中序中线性查找是 O(n)
 *    - 总复杂度会变成 O(n²)
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 从中序与后序遍历序列构造二叉树 → 后序最后一个是根
 * - 根据前序和后序遍历构造二叉树 → 需要特殊处理
 * - 构造最大二叉树 → 分治思想
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 数据还原：从序列化数据还原树结构
 * 2. 编译原理：从 token 序列构造 AST
 */

export { TreeNode, buildTree };
export default buildTree;

