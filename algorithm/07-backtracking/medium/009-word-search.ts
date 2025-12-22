/**
 * ============================================================
 * LeetCode 79. 单词搜索 (Word Search)
 * ============================================================
 *
 * 题目描述：
 * 给定一个 m x n 二维字符网格 board 和一个字符串单词 word。
 * 如果 word 存在于网格中，返回 true；否则，返回 false。
 *
 * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，
 * 其中"相邻"单元格是那些水平相邻或垂直相邻的单元格。
 * 同一个单元格内的字母不允许被重复使用。
 *
 * 示例 1：
 * 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
 * 输出：true
 *
 * 示例 2：
 * 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
 * 输出：true
 *
 * 示例 3：
 * 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
 * 输出：false
 *
 * 提示：
 * m == board.length
 * n = board[i].length
 * 1 <= m, n <= 6
 * 1 <= word.length <= 15
 * board 和 word 仅由大小写英文字母组成
 */

// ============================================================
// 方法一：DFS + 回溯（原地标记）
// ============================================================

/**
 * 📊 核心思路：
 *
 * 1. 遍历每个格子作为起点
 * 2. 从起点开始 DFS，尝试匹配 word
 * 3. 匹配成功继续向四个方向探索
 * 4. 使用原地修改标记已访问，回溯时恢复
 *
 * 🔄 DFS 过程 (Mermaid):
 *
 * ```mermaid
 * flowchart TD
 *     A[开始DFS] --> B{边界检查}
 *     B -->|越界| C[返回false]
 *     B -->|OK| D{字符匹配?}
 *     D -->|不匹配| E[返回false]
 *     D -->|匹配| F{是最后一个字符?}
 *     F -->|是| G[返回true]
 *     F -->|否| H[标记已访问]
 *     H --> I[向四个方向DFS]
 *     I --> J{任一方向成功?}
 *     J -->|是| K[返回true]
 *     J -->|否| L[恢复标记]
 *     L --> M[返回false]
 * ```
 */
function exist(board: string[][], word: string): boolean {
  const rows = board.length;
  const cols = board[0].length;

  /**
   * DFS 搜索
   * @param i 当前行
   * @param j 当前列
   * @param k word 的第 k 个字符
   */
  function dfs(i: number, j: number, k: number): boolean {
    // 1. 边界检查
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
      return false;
    }

    // 2. 字符不匹配
    if (board[i][j] !== word[k]) {
      return false;
    }

    // 3. 找到了完整的单词
    if (k === word.length - 1) {
      return true;
    }

    // 4. 标记当前格子已访问（原地修改）
    const temp = board[i][j];
    board[i][j] = '#';

    // 5. 向四个方向探索
    const found =
      dfs(i + 1, j, k + 1) ||
      dfs(i - 1, j, k + 1) ||
      dfs(i, j + 1, k + 1) ||
      dfs(i, j - 1, k + 1);

    // 6. 恢复标记（回溯）
    board[i][j] = temp;

    return found;
  }

  // 遍历每个格子作为起点
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (dfs(i, j, 0)) {
        return true;
      }
    }
  }

  return false;
}

// ============================================================
// 方法二：使用方向数组（更清晰的写法）
// ============================================================

/**
 * 📊 使用方向数组的好处：
 * - 代码更清晰
 * - 容易扩展到八方向
 */
function existWithDirections(board: string[][], word: string): boolean {
  const rows = board.length;
  const cols = board[0].length;
  const directions = [
    [0, 1],  // 右
    [0, -1], // 左
    [1, 0],  // 下
    [-1, 0], // 上
  ];

  function dfs(i: number, j: number, k: number): boolean {
    // 字符不匹配
    if (board[i][j] !== word[k]) {
      return false;
    }

    // 找到了
    if (k === word.length - 1) {
      return true;
    }

    // 标记
    const temp = board[i][j];
    board[i][j] = '#';

    // 四个方向
    for (const [di, dj] of directions) {
      const ni = i + di;
      const nj = j + dj;

      // 边界检查
      if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
        if (dfs(ni, nj, k + 1)) {
          board[i][j] = temp; // 记得恢复
          return true;
        }
      }
    }

    // 恢复
    board[i][j] = temp;
    return false;
  }

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (dfs(i, j, 0)) {
        return true;
      }
    }
  }

  return false;
}

// ============================================================
// 方法三：使用 visited 数组（不修改原数组）
// ============================================================

/**
 * 📊 如果不能修改原数组，使用 visited 数组
 */
function existWithVisited(board: string[][], word: string): boolean {
  const rows = board.length;
  const cols = board[0].length;
  const visited: boolean[][] = Array.from({ length: rows }, () =>
    new Array(cols).fill(false)
  );

  function dfs(i: number, j: number, k: number): boolean {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
      return false;
    }

    if (visited[i][j] || board[i][j] !== word[k]) {
      return false;
    }

    if (k === word.length - 1) {
      return true;
    }

    visited[i][j] = true;

    const found =
      dfs(i + 1, j, k + 1) ||
      dfs(i - 1, j, k + 1) ||
      dfs(i, j + 1, k + 1) ||
      dfs(i, j - 1, k + 1);

    visited[i][j] = false;

    return found;
  }

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (dfs(i, j, 0)) {
        return true;
      }
    }
  }

  return false;
}

// ============================================================
// 方法四：带剪枝优化
// ============================================================

/**
 * 📊 剪枝优化：
 * 1. 预先统计字符频率，如果 word 中某字符的数量超过 board，直接返回 false
 * 2. 如果 word 首字符在 board 中出现次数少于尾字符，反转 word
 */
function existOptimized(board: string[][], word: string): boolean {
  const rows = board.length;
  const cols = board[0].length;

  // 统计 board 中每个字符的数量
  const boardCount = new Map<string, number>();
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      boardCount.set(board[i][j], (boardCount.get(board[i][j]) || 0) + 1);
    }
  }

  // 统计 word 中每个字符的数量
  const wordCount = new Map<string, number>();
  for (const char of word) {
    wordCount.set(char, (wordCount.get(char) || 0) + 1);
  }

  // 剪枝1：检查每个字符是否足够
  for (const [char, count] of wordCount) {
    if ((boardCount.get(char) || 0) < count) {
      return false;
    }
  }

  // 剪枝2：如果首字符出现次数 > 尾字符，反转 word
  const firstCount = boardCount.get(word[0]) || 0;
  const lastCount = boardCount.get(word[word.length - 1]) || 0;
  if (firstCount > lastCount) {
    word = word.split('').reverse().join('');
  }

  // 标准 DFS
  function dfs(i: number, j: number, k: number): boolean {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
      return false;
    }

    if (board[i][j] !== word[k]) {
      return false;
    }

    if (k === word.length - 1) {
      return true;
    }

    const temp = board[i][j];
    board[i][j] = '#';

    const found =
      dfs(i + 1, j, k + 1) ||
      dfs(i - 1, j, k + 1) ||
      dfs(i, j + 1, k + 1) ||
      dfs(i, j - 1, k + 1);

    board[i][j] = temp;
    return found;
  }

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (dfs(i, j, 0)) {
        return true;
      }
    }
  }

  return false;
}

// ============================================================
// 📊 复杂度分析
// ============================================================

/**
 * 时间复杂度：O(m * n * 3^L)
 * - m * n 是 board 的大小，遍历每个起点
 * - 每个起点最多搜索 3^L 个路径（L 是 word 长度）
 * - 为什么是 3^L 而不是 4^L？因为不能回到刚来的方向
 *
 * 空间复杂度：O(L)
 * - 递归栈深度最大为 word 长度
 * - 如果使用 visited 数组，额外 O(m * n)
 */

// ============================================================
// 🔍 图解示例
// ============================================================

/**
 * 示例：board = [["A","B","C","E"],
 *                ["S","F","C","S"],
 *                ["A","D","E","E"]]
 *        word = "ABCCED"
 *
 * 搜索过程：
 *
 *   A → B → C → C → E → D
 *   ↓                   ↑
 *   从(0,0)开始，向右→下→下→左→上
 *
 *   [A][B][C] E
 *    S  F [C] S
 *    A [D][E] E
 *
 * 步骤：
 * 1. 从 A(0,0) 开始，匹配 word[0]='A' ✓
 * 2. 向右 B(0,1)，匹配 word[1]='B' ✓
 * 3. 向右 C(0,2)，匹配 word[2]='C' ✓
 * 4. 向下 C(1,2)，匹配 word[3]='C' ✓
 * 5. 向下 E(2,2)，匹配 word[4]='E' ✓
 * 6. 向左 D(2,1)，匹配 word[5]='D' ✓
 * 7. 完成！返回 true
 */

// ============================================================
// ⚠️ 易错点
// ============================================================

/**
 * 1. 回溯时必须恢复标记
 *    - 错误：忘记 board[i][j] = temp
 *    - 这会导致后续搜索无法使用这个格子
 *
 * 2. 提前返回优化
 *    - 找到后应该立即返回 true，不要继续搜索
 *    - 使用 || 短路求值
 *
 * 3. 边界检查顺序
 *    - 先检查边界，再检查字符
 *    - 否则会数组越界
 *
 * 4. 原地修改注意
 *    - 如果是面试，问清楚能否修改原数组
 *    - 不能的话用 visited 数组
 */

// ============================================================
// 🔗 举一反三
// ============================================================

/**
 * 单词搜索 II (LeetCode 212) - Hard
 * - 给定多个单词，找出所有存在于 board 中的
 * - 优化：使用 Trie（字典树）
 *
 * 岛屿数量 (LeetCode 200)
 * - 二维 DFS 的另一个经典题
 * - 区别：找连通区域，不是找路径
 *
 * 矩阵中的路径（剑指 Offer 12）
 * - 同一道题
 */

// ============================================================
// 测试用例
// ============================================================

function test() {
  const board1 = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E'],
  ];

  console.log(exist([...board1.map((r) => [...r])], 'ABCCED')); // true
  console.log(exist([...board1.map((r) => [...r])], 'SEE')); // true
  console.log(exist([...board1.map((r) => [...r])], 'ABCB')); // false

  console.log(
    existOptimized([...board1.map((r) => [...r])], 'ABCCED')
  ); // true
}

// test();

export { exist, existWithDirections, existWithVisited, existOptimized };

