/**
 * 📝 题目：搜索二维矩阵
 * 🔗 链接：https://leetcode.cn/problems/search-a-2d-matrix/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：数组、二分查找、矩阵
 *
 * 📋 题目描述：
 * 给你一个满足下述两条属性的 m x n 整数矩阵：
 * - 每行中的整数从左到右按非递减顺序排列。
 * - 每行的第一个整数大于前一行的最后一个整数。
 *
 * 给你一个整数 target，如果 target 在矩阵中，返回 true；否则，返回 false。
 *
 * 示例：
 * 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
 * 输出：true
 *
 *  ┌───┬───┬───┬───┐
 *  │ 1 │ 3 │ 5 │ 7 │
 *  ├───┼───┼───┼───┤
 *  │10 │11 │16 │20 │
 *  ├───┼───┼───┼───┤
 *  │23 │30 │34 │60 │
 *  └───┴───┴───┴───┘
 */

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 矩阵的特性决定了它可以看作一个一维有序数组！
//
// 二维坐标 (i, j) 和一维索引 k 的转换：
// - k = i * n + j
// - i = Math.floor(k / n)
// - j = k % n
//
// 所以可以直接用标准二分查找！

// ============================================================
// 解法一：一维二分
// ============================================================
// ⏱️ 时间复杂度：O(log(mn)) | 空间复杂度：O(1)

/**
 * 📊 一维二分图解：
 *
 * 矩阵展开为一维：
 * [1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60]
 *  0  1  2  3   4   5   6   7   8   9  10  11
 *
 * 索引 k 对应的矩阵位置：
 * - 行 i = k / 4
 * - 列 j = k % 4
 */
function searchMatrix(matrix: number[][], target: number): boolean {
  const m = matrix.length;
  const n = matrix[0].length;

  let left = 0;
  let right = m * n - 1;

  while (left <= right) {
    const mid = left + ((right - left) >> 1);
    // 一维索引转二维坐标
    const midVal = matrix[Math.floor(mid / n)][mid % n];

    if (midVal === target) {
      return true;
    } else if (midVal < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return false;
}

// ============================================================
// 解法二：两次二分
// ============================================================
// ⏱️ 时间复杂度：O(log m + log n) | 空间复杂度：O(1)

/**
 * 1. 先二分找到 target 可能在哪一行
 * 2. 再在那一行中二分查找 target
 */
function searchMatrix_v2(matrix: number[][], target: number): boolean {
  const m = matrix.length;
  const n = matrix[0].length;

  // 二分找行：找最后一个首元素 <= target 的行
  let top = 0;
  let bottom = m - 1;

  while (top < bottom) {
    const mid = top + ((bottom - top + 1) >> 1);
    if (matrix[mid][0] <= target) {
      top = mid;
    } else {
      bottom = mid - 1;
    }
  }

  const row = top;

  // 在该行中二分查找
  let left = 0;
  let right = n - 1;

  while (left <= right) {
    const mid = left + ((right - left) >> 1);
    if (matrix[row][mid] === target) {
      return true;
    } else if (matrix[row][mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return false;
}

// ============================================================
// 解法三：从右上角开始搜索
// ============================================================
// ⏱️ 时间复杂度：O(m + n) | 空间复杂度：O(1)

/**
 * 从右上角开始：
 * - 如果当前值 > target，往左走
 * - 如果当前值 < target，往下走
 * - 如果当前值 == target，找到了
 */
function searchMatrix_v3(matrix: number[][], target: number): boolean {
  const m = matrix.length;
  const n = matrix[0].length;

  let row = 0;
  let col = n - 1;

  while (row < m && col >= 0) {
    if (matrix[row][col] === target) {
      return true;
    } else if (matrix[row][col] > target) {
      col--;
    } else {
      row++;
    }
  }

  return false;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法       | 时间         | 空间  | 特点                  |
 * |-----------|--------------|-------|----------------------|
 * | 一维二分   | O(log(mn))   | O(1)  | 推荐，代码简洁         |
 * | 两次二分   | O(log m+log n)| O(1)  | 同样是对数级           |
 * | 右上角搜索 | O(m+n)       | O(1)  | 不是二分，但也很高效   |
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 坐标转换：
 *    - i = k / n，j = k % n
 *    - 不要搞反行和列
 *
 * 2. 矩阵特性：
 *    - 这道题的矩阵是严格有序的（可以展开为一维）
 *    - "搜索二维矩阵 II" 的矩阵不能这样处理
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 搜索二维矩阵 II → 只保证行列有序，不能展开为一维
 * - 有序矩阵中第 K 小的元素 → 更复杂
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 表格搜索：在有序表格数据中快速定位
 * 2. 日历查找：在日期矩阵中查找特定日期
 */

export { searchMatrix, searchMatrix_v2, searchMatrix_v3 };
export default searchMatrix;

