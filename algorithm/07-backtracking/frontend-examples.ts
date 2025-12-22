/**
 * ============================================================
 * 📚 回溯算法 - 前端业务场景代码示例
 * ============================================================
 *
 * 本文件展示回溯算法在前端实际业务中的应用
 */

// ============================================================
// 1. 满减凑单（组合总和）
// ============================================================

/**
 * 📝 业务场景：电商满减凑单
 *
 * 场景描述：
 * - 用户购物车中有多个商品
 * - 满 X 元减 Y 元
 * - 找出所有能凑到满减门槛的商品组合
 */
interface Product {
  id: string;
  name: string;
  price: number;
}

interface CombinationResult {
  products: Product[];
  totalPrice: number;
  savings: number;
}

class DiscountCombiner {
  /**
   * 找出所有能达到满减门槛的商品组合
   */
  findCombinations(
    products: Product[],
    threshold: number,
    discount: number,
    maxOverflow: number = 50 // 最多超过门槛多少
  ): CombinationResult[] {
    const results: CombinationResult[] = [];
    const path: Product[] = [];

    // 按价格排序，便于剪枝
    const sorted = [...products].sort((a, b) => a.price - b.price);

    const backtrack = (start: number, currentSum: number) => {
      // 达到门槛且不超太多
      if (currentSum >= threshold && currentSum <= threshold + maxOverflow) {
        results.push({
          products: [...path],
          totalPrice: currentSum,
          savings: discount,
        });
      }

      // 超过太多就剪枝
      if (currentSum > threshold + maxOverflow) {
        return;
      }

      for (let i = start; i < sorted.length; i++) {
        path.push(sorted[i]);
        backtrack(i + 1, currentSum + sorted[i].price);
        path.pop();
      }
    };

    backtrack(0, 0);

    // 按总价排序，优先推荐接近门槛的
    return results.sort((a, b) => a.totalPrice - b.totalPrice);
  }
}

// ============================================================
// 2. 权限组合生成
// ============================================================

/**
 * 📝 业务场景：RBAC 权限配置
 *
 * 场景描述：
 * - 生成所有可能的权限组合
 * - 用于权限模板创建
 */
type Permission = 'read' | 'write' | 'delete' | 'admin';

interface PermissionSet {
  permissions: Permission[];
  level: string;
}

class PermissionGenerator {
  private permissions: Permission[] = ['read', 'write', 'delete', 'admin'];

  /**
   * 生成所有权限子集
   */
  generateAllSets(): PermissionSet[] {
    const results: PermissionSet[] = [];
    const path: Permission[] = [];

    const backtrack = (start: number) => {
      // 每个节点都是一个有效的权限组合
      results.push({
        permissions: [...path],
        level: this.getLevel(path),
      });

      for (let i = start; i < this.permissions.length; i++) {
        path.push(this.permissions[i]);
        backtrack(i + 1);
        path.pop();
      }
    };

    backtrack(0);
    return results;
  }

  private getLevel(permissions: Permission[]): string {
    if (permissions.includes('admin')) return '管理员';
    if (permissions.includes('delete')) return '高级用户';
    if (permissions.includes('write')) return '普通用户';
    if (permissions.includes('read')) return '只读用户';
    return '无权限';
  }
}

// ============================================================
// 3. 表单字段排列
// ============================================================

/**
 * 📝 业务场景：动态表单生成
 *
 * 场景描述：
 * - 用户可以自定义表单字段顺序
 * - 生成所有可能的排列供预览
 */
interface FormField {
  id: string;
  label: string;
  type: 'input' | 'select' | 'textarea';
}

class FormLayoutGenerator {
  /**
   * 生成所有字段排列
   */
  generateLayouts(fields: FormField[]): FormField[][] {
    const results: FormField[][] = [];
    const path: FormField[] = [];
    const used: boolean[] = new Array(fields.length).fill(false);

    const backtrack = () => {
      if (path.length === fields.length) {
        results.push([...path]);
        return;
      }

      for (let i = 0; i < fields.length; i++) {
        if (used[i]) continue;

        path.push(fields[i]);
        used[i] = true;

        backtrack();

        path.pop();
        used[i] = false;
      }
    };

    backtrack();
    return results;
  }

  /**
   * 生成前 N 个排列（限制结果数量）
   */
  generateTopLayouts(fields: FormField[], limit: number): FormField[][] {
    const results: FormField[][] = [];
    const path: FormField[] = [];
    const used: boolean[] = new Array(fields.length).fill(false);

    const backtrack = (): boolean => {
      if (path.length === fields.length) {
        results.push([...path]);
        return results.length >= limit;
      }

      for (let i = 0; i < fields.length; i++) {
        if (used[i]) continue;

        path.push(fields[i]);
        used[i] = true;

        if (backtrack()) return true;

        path.pop();
        used[i] = false;
      }

      return false;
    };

    backtrack();
    return results;
  }
}

// ============================================================
// 4. 迷宫求解
// ============================================================

/**
 * 📝 业务场景：游戏路径规划
 *
 * 场景描述：
 * - 二维迷宫中找到从起点到终点的所有路径
 * - 适用于游戏、地图导航等
 */
type Cell = 0 | 1; // 0 可通行，1 障碍物
type Direction = 'up' | 'down' | 'left' | 'right';

interface Position {
  row: number;
  col: number;
}

interface PathResult {
  path: Position[];
  directions: Direction[];
}

class MazeSolver {
  private maze: Cell[][];
  private rows: number;
  private cols: number;
  private directions: [number, number, Direction][] = [
    [-1, 0, 'up'],
    [1, 0, 'down'],
    [0, -1, 'left'],
    [0, 1, 'right'],
  ];

  constructor(maze: Cell[][]) {
    this.maze = maze;
    this.rows = maze.length;
    this.cols = maze[0]?.length || 0;
  }

  /**
   * 找到所有路径
   */
  findAllPaths(start: Position, end: Position): PathResult[] {
    const results: PathResult[] = [];
    const path: Position[] = [];
    const dirs: Direction[] = [];
    const visited: boolean[][] = Array.from({ length: this.rows }, () =>
      new Array(this.cols).fill(false)
    );

    const backtrack = (row: number, col: number) => {
      // 到达终点
      if (row === end.row && col === end.col) {
        results.push({
          path: [...path, { row, col }],
          directions: [...dirs],
        });
        return;
      }

      path.push({ row, col });
      visited[row][col] = true;

      for (const [dr, dc, dir] of this.directions) {
        const newRow = row + dr;
        const newCol = col + dc;

        if (this.isValid(newRow, newCol, visited)) {
          dirs.push(dir);
          backtrack(newRow, newCol);
          dirs.pop();
        }
      }

      path.pop();
      visited[row][col] = false;
    };

    if (this.isValid(start.row, start.col, visited)) {
      backtrack(start.row, start.col);
    }

    return results;
  }

  private isValid(row: number, col: number, visited: boolean[][]): boolean {
    return (
      row >= 0 &&
      row < this.rows &&
      col >= 0 &&
      col < this.cols &&
      this.maze[row][col] === 0 &&
      !visited[row][col]
    );
  }

  /**
   * 找最短路径（BFS 更合适，但这里展示回溯思路）
   */
  findShortestPath(start: Position, end: Position): PathResult | null {
    const allPaths = this.findAllPaths(start, end);
    if (allPaths.length === 0) return null;

    return allPaths.reduce((shortest, current) =>
      current.path.length < shortest.path.length ? current : shortest
    );
  }
}

// ============================================================
// 5. URL 路径分割
// ============================================================

/**
 * 📝 业务场景：路由解析
 *
 * 场景描述：
 * - 将 URL 路径分割成多种可能的组合
 * - 用于路由匹配、面包屑生成等
 */
class PathSplitter {
  /**
   * 获取路径的所有分割方式
   * 例如 "/a/b/c" 可以分割为 ["/a", "/b", "/c"] 或 ["/a/b", "/c"] 等
   */
  splitPath(path: string): string[][] {
    const segments = path.split('/').filter(Boolean);
    const results: string[][] = [];
    const current: string[] = [];

    const backtrack = (start: number) => {
      if (start === segments.length) {
        results.push([...current]);
        return;
      }

      // 尝试从 start 开始的每种长度
      for (let end = start; end < segments.length; end++) {
        const segment = '/' + segments.slice(start, end + 1).join('/');
        current.push(segment);
        backtrack(end + 1);
        current.pop();
      }
    };

    backtrack(0);
    return results;
  }
}

// ============================================================
// 6. 工作日排班
// ============================================================

/**
 * 📝 业务场景：员工排班系统
 *
 * 场景描述：
 * - 从可用员工中选择 K 人排班
 * - 考虑员工偏好和约束
 */
interface Employee {
  id: string;
  name: string;
  preferredDays: number[]; // 偏好的工作日 1-7
}

interface Schedule {
  employees: Employee[];
  day: number;
}

class SchedulePlanner {
  /**
   * 生成某天的所有可能排班组合
   */
  generateSchedules(
    employees: Employee[],
    day: number,
    requiredCount: number
  ): Schedule[] {
    // 过滤出偏好这天的员工
    const available = employees.filter((e) =>
      e.preferredDays.includes(day)
    );

    const results: Schedule[] = [];
    const path: Employee[] = [];

    const backtrack = (start: number) => {
      if (path.length === requiredCount) {
        results.push({
          employees: [...path],
          day,
        });
        return;
      }

      // 剪枝：剩余人数不够
      if (available.length - start < requiredCount - path.length) {
        return;
      }

      for (let i = start; i < available.length; i++) {
        path.push(available[i]);
        backtrack(i + 1);
        path.pop();
      }
    };

    backtrack(0);
    return results;
  }
}

// ============================================================
// 7. 标签多选组合
// ============================================================

/**
 * 📝 业务场景：标签筛选器
 *
 * 场景描述：
 * - 用户可以选择多个标签进行筛选
 * - 显示所有可能的筛选组合及对应结果数
 */
interface Tag {
  id: string;
  name: string;
  count: number;
}

interface FilterCombination {
  tags: Tag[];
  expectedCount: number;
}

class TagFilterGenerator {
  /**
   * 生成所有标签组合
   */
  generateCombinations(
    tags: Tag[],
    minTags: number = 1,
    maxTags: number = Infinity
  ): FilterCombination[] {
    const results: FilterCombination[] = [];
    const path: Tag[] = [];

    const backtrack = (start: number) => {
      if (path.length >= minTags && path.length <= maxTags) {
        results.push({
          tags: [...path],
          expectedCount: this.estimateCount(path),
        });
      }

      if (path.length >= maxTags) return;

      for (let i = start; i < tags.length; i++) {
        path.push(tags[i]);
        backtrack(i + 1);
        path.pop();
      }
    };

    backtrack(0);
    return results;
  }

  private estimateCount(selectedTags: Tag[]): number {
    // 简化估算：取最小的 count
    if (selectedTags.length === 0) return 0;
    return Math.min(...selectedTags.map((t) => t.count));
  }
}

// ============================================================
// 8. 括号生成
// ============================================================

/**
 * 📝 业务场景：代码生成器
 *
 * 场景描述：
 * - 生成所有有效的括号组合
 * - 用于代码模板生成、语法高亮等
 */
class ParenthesisGenerator {
  /**
   * 生成 n 对有效括号的所有组合
   */
  generate(n: number): string[] {
    const results: string[] = [];

    const backtrack = (current: string, open: number, close: number) => {
      if (current.length === n * 2) {
        results.push(current);
        return;
      }

      // 可以添加左括号
      if (open < n) {
        backtrack(current + '(', open + 1, close);
      }

      // 可以添加右括号（只有当右括号数 < 左括号数时）
      if (close < open) {
        backtrack(current + ')', open, close + 1);
      }
    };

    backtrack('', 0, 0);
    return results;
  }

  /**
   * 生成带自定义括号类型的组合
   */
  generateCustom(
    n: number,
    openChar: string,
    closeChar: string
  ): string[] {
    const results: string[] = [];

    const backtrack = (current: string, open: number, close: number) => {
      if (current.length === n * 2) {
        results.push(current);
        return;
      }

      if (open < n) {
        backtrack(current + openChar, open + 1, close);
      }

      if (close < open) {
        backtrack(current + closeChar, open, close + 1);
      }
    };

    backtrack('', 0, 0);
    return results;
  }
}

// ============================================================
// 导出
// ============================================================

export {
  DiscountCombiner,
  PermissionGenerator,
  FormLayoutGenerator,
  MazeSolver,
  PathSplitter,
  SchedulePlanner,
  TagFilterGenerator,
  ParenthesisGenerator,
};

export type {
  Product,
  CombinationResult,
  Permission,
  PermissionSet,
  FormField,
  Cell,
  Direction,
  Position,
  PathResult,
  Employee,
  Schedule,
  Tag,
  FilterCombination,
};

