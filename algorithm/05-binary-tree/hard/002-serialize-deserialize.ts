/**
 * 📝 题目：二叉树的序列化与反序列化
 * 🔗 链接：https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/
 * 🏷️ 难度：Hard
 * 🏷️ 标签：树、深度优先搜索、广度优先搜索、设计、字符串、二叉树
 *
 * 📋 题目描述：
 * 序列化是将一个数据结构或者对象转换为连续的比特位的操作，
 * 进而可以将转换后的数据存储在一个文件或者内存中，
 * 同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。
 *
 * 请设计一个算法来实现二叉树的序列化与反序列化。
 * 这里不限定你的序列 / 反序列化算法执行逻辑，
 * 你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。
 *
 * 示例：
 *        1
 *       / \
 *      2   3
 *         / \
 *        4   5
 *
 * 输入：root = [1,2,3,null,null,4,5]
 * 输出：[1,2,3,null,null,4,5]
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
// 方法一：前序遍历（DFS）
// - 序列化：前序遍历，空节点用特殊标记
// - 反序列化：按前序顺序重建
//
// 方法二：层序遍历（BFS）
// - 序列化：层序遍历，记录所有节点包括 null
// - 反序列化：层序重建
//
// 为什么单独的前序/中序/后序不行？
// - 因为无法区分左右子树的边界
// - 需要额外标记空节点

// ============================================================
// 解法一：前序遍历（DFS）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n)

/**
 * 📊 前序序列化图解：
 *
 *        1
 *       / \
 *      2   3
 *         / \
 *        4   5
 *
 * 前序遍历（标记空节点为 #）：
 * 1 → 2 → # → # → 3 → 4 → # → # → 5 → # → #
 *
 * 序列化结果：\"1,2,#,#,3,4,#,#,5,#,#\"
 *
 * 反序列化：
 * 读取 1 → 创建节点 1
 * 递归创建左子树
 *   读取 2 → 创建节点 2
 *   递归创建左子树 → 读取 # → 返回 null
 *   递归创建右子树 → 读取 # → 返回 null
 *   返回节点 2
 * 递归创建右子树
 *   读取 3 → 创建节点 3
 *   ... 继续递归
 */

const NULL_MARKER = '#';
const SEPARATOR = ',';

function serialize(root: TreeNode | null): string {
  const result: string[] = [];

  function preorder(node: TreeNode | null) {
    if (!node) {
      result.push(NULL_MARKER);
      return;
    }

    result.push(String(node.val));
    preorder(node.left);
    preorder(node.right);
  }

  preorder(root);
  return result.join(SEPARATOR);
}

function deserialize(data: string): TreeNode | null {
  const values = data.split(SEPARATOR);
  let index = 0;

  function buildTree(): TreeNode | null {
    if (index >= values.length) return null;

    const val = values[index++];

    if (val === NULL_MARKER) {
      return null;
    }

    const node = new TreeNode(parseInt(val, 10));
    node.left = buildTree();
    node.right = buildTree();

    return node;
  }

  return buildTree();
}

// ============================================================
// 解法二：层序遍历（BFS）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n)

/**
 * 📊 层序序列化图解：
 *
 *        1
 *       / \
 *      2   3
 *         / \
 *        4   5
 *
 * 层序遍历：1, 2, 3, #, #, 4, 5, #, #, #, #
 */
function serialize_bfs(root: TreeNode | null): string {
  if (!root) return NULL_MARKER;

  const result: string[] = [];
  const queue: Array<TreeNode | null> = [root];

  while (queue.length > 0) {
    const node = queue.shift();

    if (node) {
      result.push(String(node.val));
      queue.push(node.left);
      queue.push(node.right);
    } else {
      result.push(NULL_MARKER);
    }
  }

  // 去掉末尾的 null
  while (result[result.length - 1] === NULL_MARKER) {
    result.pop();
  }

  return result.join(SEPARATOR);
}

function deserialize_bfs(data: string): TreeNode | null {
  if (data === NULL_MARKER || data === '') return null;

  const values = data.split(SEPARATOR);
  const root = new TreeNode(parseInt(values[0], 10));
  const queue: TreeNode[] = [root];
  let index = 1;

  while (queue.length > 0 && index < values.length) {
    const node = queue.shift()!;

    // 左子节点
    if (index < values.length && values[index] !== NULL_MARKER) {
      node.left = new TreeNode(parseInt(values[index], 10));
      queue.push(node.left);
    }
    index++;

    // 右子节点
    if (index < values.length && values[index] !== NULL_MARKER) {
      node.right = new TreeNode(parseInt(values[index], 10));
      queue.push(node.right);
    }
    index++;
  }

  return root;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法     | 时间  | 空间  | 特点                     |
 * |---------|-------|-------|-------------------------|
 * | DFS     | O(n)  | O(n)  | 推荐，代码简洁            |
 * | BFS     | O(n)  | O(n)  | 更直观，与数组表示一致    |
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 空节点必须标记：
 *    - 不标记空节点无法确定树的结构
 *
 * 2. 分隔符选择：
 *    - 如果节点值可能是负数，要注意分隔符
 *
 * 3. BFS 末尾 null 处理：
 *    - 可以去掉也可以保留，但反序列化要一致
 *
 * 4. 递归中的索引：
 *    - DFS 用闭包变量或传引用
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 二叉搜索树的序列化与反序列化 → 利用 BST 性质
 * - N 叉树的序列化与反序列化 → 多个子节点
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 数据持久化：将树结构保存到 localStorage
 * 2. 网络传输：通过 API 传输树结构
 * 3. 深拷贝：序列化后反序列化实现深拷贝
 * 4. 撤销/重做：保存和恢复树状态
 */

export {
  TreeNode,
  serialize,
  deserialize,
  serialize_bfs,
  deserialize_bfs,
};

