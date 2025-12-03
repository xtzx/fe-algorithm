/**
 * ============================================================
 * 📚 Step 05: 二叉树
 * ============================================================
 *
 * 本章内容：
 * 1. 二叉树基础（结构定义、术语）
 * 2. 遍历方式（前中后序、层序）
 * 3. 递归思想（分解问题）
 * 4. DFS 与 BFS
 * 5. 常见题型模板
 */

// ============================================================
// 1. 二叉树基础
// ============================================================

/**
 * 📖 二叉树结构图解：
 *
 *           1        ← 根节点 (root)
 *          / \
 *         2   3      ← 层级 1
 *        / \   \
 *       4   5   6    ← 叶子节点 (leaf)
 *
 * 术语：
 * - 根节点 (root): 最顶层的节点
 * - 叶子节点 (leaf): 没有子节点的节点
 * - 深度 (depth): 从根到该节点的边数
 * - 高度 (height): 从该节点到最远叶子的边数
 * - 层级 (level): 深度 + 1
 *
 * 特殊二叉树：
 * - 满二叉树: 每个节点要么有0个子节点，要么有2个
 * - 完全二叉树: 除最后一层外都是满的，最后一层左对齐
 * - 平衡二叉树: 任意节点左右子树高度差 ≤ 1
 * - 二叉搜索树 (BST): 左子树 < 根 < 右子树
 */

// ────────────────────────────────────────────────────────────
// 二叉树节点定义
// ────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────
// 辅助函数：从数组创建二叉树（层序）
// ────────────────────────────────────────────────────────────

function createBinaryTree(arr: (number | null)[]): TreeNode | null {
  if (arr.length === 0 || arr[0] === null) return null;

  const root = new TreeNode(arr[0]);
  const queue: TreeNode[] = [root];
  let i = 1;

  while (queue.length > 0 && i < arr.length) {
    const node = queue.shift()!;

    if (i < arr.length && arr[i] !== null) {
      node.left = new TreeNode(arr[i] as number);
      queue.push(node.left);
    }
    i++;

    if (i < arr.length && arr[i] !== null) {
      node.right = new TreeNode(arr[i] as number);
      queue.push(node.right);
    }
    i++;
  }

  return root;
}

// ============================================================
// 2. 🎯 二叉树 - 适用场景与信号词
// ============================================================

/**
 * 🎯 信号词识别：
 *
 * ┌────────────────────────────────────────────────────────────┐
 * │                   二叉树问题分类                            │
 * ├────────────────────────────────────────────────────────────┤
 * │                                                            │
 * │  【遍历类】                                                 │
 * │   信号词: 遍历、前序、中序、后序、层序                      │
 * │   典型题: 各种遍历、层序遍历、之字形遍历                    │
 * │                                                            │
 * │  【属性类】                                                 │
 * │   信号词: 深度、高度、节点数、是否对称/相同/平衡            │
 * │   典型题: 最大深度、是否平衡、对称二叉树                    │
 * │                                                            │
 * │  【路径类】                                                 │
 * │   信号词: 路径、路径和、最长路径                            │
 * │   典型题: 路径总和、二叉树直径、最大路径和                  │
 * │                                                            │
 * │  【构造类】                                                 │
 * │   信号词: 构造、还原、根据遍历序列                          │
 * │   典型题: 从前序+中序构造、从有序数组构造BST                │
 * │                                                            │
 * │  【BST类】                                                  │
 * │   信号词: 搜索、插入、删除、第K大/小                        │
 * │   典型题: 验证BST、BST中第K小、BST公共祖先                  │
 * │                                                            │
 * │  【公共祖先】                                               │
 * │   信号词: 最近公共祖先、LCA                                 │
 * │   典型题: 二叉树的最近公共祖先                              │
 * │                                                            │
 * └────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 3. 代码模板
// ============================================================

// ────────────────────────────────────────────────────────────
// 模板一：前中后序遍历（递归）
// ────────────────────────────────────────────────────────────

/**
 * 📊 遍历顺序图解：
 *
 *        1
 *       / \
 *      2   3
 *     / \
 *    4   5
 *
 *  前序 (Pre-order):  根 → 左 → 右  →  [1, 2, 4, 5, 3]
 *  中序 (In-order):   左 → 根 → 右  →  [4, 2, 5, 1, 3]
 *  后序 (Post-order): 左 → 右 → 根  →  [4, 5, 2, 3, 1]
 *
 * 记忆技巧：
 * - 前/中/后 指的是 "根" 的位置
 * - 左右顺序始终是 左 → 右
 */

// 前序遍历：根 → 左 → 右
function preorderTraversal(root: TreeNode | null): number[] {
  const result: number[] = [];

  function traverse(node: TreeNode | null) {
    if (!node) return;

    result.push(node.val); // 根
    traverse(node.left); // 左
    traverse(node.right); // 右
  }

  traverse(root);
  return result;
}

// 中序遍历：左 → 根 → 右
function inorderTraversal(root: TreeNode | null): number[] {
  const result: number[] = [];

  function traverse(node: TreeNode | null) {
    if (!node) return;

    traverse(node.left); // 左
    result.push(node.val); // 根
    traverse(node.right); // 右
  }

  traverse(root);
  return result;
}

// 后序遍历：左 → 右 → 根
function postorderTraversal(root: TreeNode | null): number[] {
  const result: number[] = [];

  function traverse(node: TreeNode | null) {
    if (!node) return;

    traverse(node.left); // 左
    traverse(node.right); // 右
    result.push(node.val); // 根
  }

  traverse(root);
  return result;
}

// ────────────────────────────────────────────────────────────
// 模板二：前中后序遍历（迭代 - 用栈）
// ────────────────────────────────────────────────────────────

/**
 * 📊 迭代遍历的关键：用栈模拟递归
 *
 * 🔄 前序遍历迭代流程 (Mermaid):
 * ```mermaid
 * flowchart TD
 *     A[根节点入栈] --> B{栈非空?}
 *     B -->|Yes| C[弹出节点, 访问]
 *     C --> D[右子节点入栈]
 *     D --> E[左子节点入栈]
 *     E --> B
 *     B -->|No| F[结束]
 * ```
 */

// 前序遍历（迭代）
function preorderIterative(root: TreeNode | null): number[] {
  if (!root) return [];

  const result: number[] = [];
  const stack: TreeNode[] = [root];

  while (stack.length > 0) {
    const node = stack.pop()!;
    result.push(node.val);

    // 先右后左入栈，这样出栈时就是先左后右
    if (node.right) stack.push(node.right);
    if (node.left) stack.push(node.left);
  }

  return result;
}

// 中序遍历（迭代）
function inorderIterative(root: TreeNode | null): number[] {
  const result: number[] = [];
  const stack: TreeNode[] = [];
  let curr: TreeNode | null = root;

  while (curr || stack.length > 0) {
    // 一直走到最左
    while (curr) {
      stack.push(curr);
      curr = curr.left;
    }

    // 访问节点
    curr = stack.pop()!;
    result.push(curr.val);

    // 转向右子树
    curr = curr.right;
  }

  return result;
}

// 后序遍历（迭代）- 使用标记法
function postorderIterative(root: TreeNode | null): number[] {
  if (!root) return [];

  const result: number[] = [];
  const stack: Array<{ node: TreeNode; visited: boolean }> = [
    { node: root, visited: false },
  ];

  while (stack.length > 0) {
    const { node, visited } = stack.pop()!;

    if (visited) {
      result.push(node.val);
    } else {
      // 后序：左右根，所以入栈顺序是：根右左
      stack.push({ node, visited: true });
      if (node.right) stack.push({ node: node.right, visited: false });
      if (node.left) stack.push({ node: node.left, visited: false });
    }
  }

  return result;
}

// ────────────────────────────────────────────────────────────
// 模板三：层序遍历（BFS）
// ────────────────────────────────────────────────────────────

/**
 * 📊 层序遍历图解：
 *
 *        1
 *       / \
 *      2   3
 *     / \   \
 *    4   5   6
 *
 *  层序遍历: [[1], [2, 3], [4, 5, 6]]
 *
 * 🔄 BFS 流程 (Mermaid):
 * ```mermaid
 * flowchart TD
 *     A[根节点入队] --> B{队列非空?}
 *     B -->|Yes| C[记录当前层大小 size]
 *     C --> D[处理 size 个节点]
 *     D --> E[每个节点: 出队, 子节点入队]
 *     E --> F[当前层结果加入结果集]
 *     F --> B
 *     B -->|No| G[返回结果]
 * ```
 */
function levelOrder(root: TreeNode | null): number[][] {
  if (!root) return [];

  const result: number[][] = [];
  const queue: TreeNode[] = [root];

  while (queue.length > 0) {
    const levelSize = queue.length;
    const currentLevel: number[] = [];

    for (let i = 0; i < levelSize; i++) {
      const node = queue.shift()!;
      currentLevel.push(node.val);

      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }

    result.push(currentLevel);
  }

  return result;
}

// ────────────────────────────────────────────────────────────
// 模板四：递归解题框架
// ────────────────────────────────────────────────────────────

/**
 * 📊 二叉树递归两种思路：
 *
 * 1. 遍历思路：不返回值，用外部变量记录结果
 *    - 像是在二叉树上"走一遍"
 *    - 适合统计、查找类问题
 *
 * 2. 分解思路：返回值，由子问题推导出原问题
 *    - 把问题分解为子问题
 *    - 适合求深度、判断结构类问题
 *
 * 🔄 递归框架 (Mermaid):
 * ```mermaid
 * flowchart TD
 *     A[递归函数] --> B{Base Case?}
 *     B -->|Yes| C[返回基本情况的结果]
 *     B -->|No| D[递归处理左子树]
 *     D --> E[递归处理右子树]
 *     E --> F[合并左右结果]
 *     F --> G[返回当前节点的结果]
 * ```
 */

// 示例：求二叉树最大深度
// 分解思路：最大深度 = max(左子树深度, 右子树深度) + 1
function maxDepth(root: TreeNode | null): number {
  if (!root) return 0;

  const leftDepth = maxDepth(root.left);
  const rightDepth = maxDepth(root.right);

  return Math.max(leftDepth, rightDepth) + 1;
}

// 示例：判断是否相同的树
// 分解思路：相同 = 根相同 && 左子树相同 && 右子树相同
function isSameTree(p: TreeNode | null, q: TreeNode | null): boolean {
  // Base Case
  if (!p && !q) return true;
  if (!p || !q) return false;

  // 当前节点值相同，且左右子树都相同
  return p.val === q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
}

// ────────────────────────────────────────────────────────────
// 模板五：最近公共祖先 (LCA)
// ────────────────────────────────────────────────────────────

/**
 * 📊 LCA 思路图解：
 *
 *        3
 *       / \
 *      5   1
 *     / \ / \
 *    6  2 0  8
 *      / \
 *     7   4
 *
 *  LCA(5, 1) = 3  （分别在左右子树）
 *  LCA(5, 4) = 5  （4 在 5 的子树中）
 *
 *  递归逻辑：
 *  - 如果当前节点是 p 或 q，返回当前节点
 *  - 递归搜索左右子树
 *  - 如果左右都找到了，当前节点就是 LCA
 *  - 如果只有一边找到，返回那一边的结果
 */
function lowestCommonAncestor(
  root: TreeNode | null,
  p: TreeNode,
  q: TreeNode
): TreeNode | null {
  // Base Case
  if (!root || root === p || root === q) {
    return root;
  }

  // 在左右子树中搜索
  const left = lowestCommonAncestor(root.left, p, q);
  const right = lowestCommonAncestor(root.right, p, q);

  // 如果左右都找到了，当前节点就是 LCA
  if (left && right) {
    return root;
  }

  // 否则返回找到的那一边
  return left || right;
}

// ============================================================
// 4. 复杂度分析
// ============================================================

/**
 * ┌─────────────────────┬────────────┬────────────┬──────────────────────┐
 * │       操作          │ 时间复杂度  │ 空间复杂度  │       说明           │
 * ├─────────────────────┼────────────┼────────────┼──────────────────────┤
 * │ 递归遍历            │   O(n)     │   O(h)     │ h 为树高，最坏 O(n)   │
 * │ 迭代遍历            │   O(n)     │   O(h)     │ 栈空间               │
 * │ 层序遍历            │   O(n)     │   O(w)     │ w 为最大层宽度       │
 * │ 求深度/高度         │   O(n)     │   O(h)     │ 需要遍历所有节点      │
 * │ 判断相同/对称       │   O(n)     │   O(h)     │ 最坏需要比较所有      │
 * │ 最近公共祖先        │   O(n)     │   O(h)     │ 最坏遍历所有节点      │
 * └─────────────────────┴────────────┴────────────┴──────────────────────┘
 *
 * 空间复杂度说明：
 * - 平衡二叉树：h = O(log n)
 * - 最坏情况（链状）：h = O(n)
 * - 层序遍历的最大宽度在完全二叉树时约为 n/2
 */

// ============================================================
// 5. ⚠️ 易错点
// ============================================================

/**
 * ┌─────────────────────────────────────────────────────────────┐
 * │                      易错点总结                              │
 * ├─────────────────────────────────────────────────────────────┤
 * │                                                             │
 * │ 【Base Case】                                                │
 * │  ⚠️ 空节点通常返回 null、0、true 等                         │
 * │  ⚠️ 叶子节点可能需要特殊处理                                │
 * │                                                             │
 * │ 【递归返回值】                                               │
 * │  ⚠️ 注意返回值的含义，子问题的定义                          │
 * │  ⚠️ 在递归回溯时处理（后序位置）还是进入时处理（前序位置）  │
 * │                                                             │
 * │ 【深度 vs 高度】                                             │
 * │  ⚠️ 深度：从根到该节点（从上往下数）                        │
 * │  ⚠️ 高度：从该节点到最远叶子（从下往上数）                  │
 * │                                                             │
 * │ 【遍历顺序】                                                 │
 * │  ⚠️ 前序：根左右，进入节点时处理                            │
 * │  ⚠️ 后序：左右根，离开节点时处理                            │
 * │  ⚠️ 中序：左根右，BST 中是有序的                            │
 * │                                                             │
 * │ 【空间复杂度】                                               │
 * │  ⚠️ 递归的空间复杂度取决于树高，不是节点数                  │
 * │  ⚠️ 最坏情况链状树 O(n)                                     │
 * │                                                             │
 * └─────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 6. 🏢 前端业务场景
// ============================================================

/**
 * ┌─────────────────────────────────────────────────────────────┐
 * │                   前端业务场景应用                           │
 * ├─────────────────────────────────────────────────────────────┤
 * │                                                             │
 * │ 【DOM 树】                                                   │
 * │  • DOM 本质上是一棵树                                       │
 * │  • querySelector 是树的搜索                                 │
 * │  • 事件冒泡/捕获是树的遍历                                  │
 * │                                                             │
 * │ 【虚拟 DOM】                                                 │
 * │  • React/Vue 的虚拟 DOM 是树结构                            │
 * │  • Diff 算法是树的比较                                      │
 * │  • 组件树的更新是树的遍历                                   │
 * │                                                             │
 * │ 【AST 抽象语法树】                                           │
 * │  • Babel 编译过程使用 AST                                   │
 * │  • ESLint 规则检查基于 AST                                  │
 * │  • 代码压缩/转换基于 AST                                    │
 * │                                                             │
 * │ 【文件系统】                                                 │
 * │  • 文件目录结构是树                                         │
 * │  • 递归读取文件夹                                           │
 * │  • 树形组件展示                                             │
 * │                                                             │
 * │ 【组织架构】                                                 │
 * │  • 人员组织树                                               │
 * │  • 菜单/导航树                                              │
 * │  • 分类层级树                                               │
 * │                                                             │
 * │ 【React Fiber】                                              │
 * │  • Fiber 节点形成的树结构                                   │
 * │  • 深度优先遍历 Fiber 树                                    │
 * │                                                             │
 * └─────────────────────────────────────────────────────────────┘
 */

export {
  TreeNode,
  createBinaryTree,
  preorderTraversal,
  inorderTraversal,
  postorderTraversal,
  preorderIterative,
  inorderIterative,
  postorderIterative,
  levelOrder,
  maxDepth,
  isSameTree,
  lowestCommonAncestor,
};

