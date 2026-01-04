/**
 * 表格数据模型
 */

// ============================================================================
// 类型定义
// ============================================================================

/** 表格行数据 */
export interface TableRow {
  id: number;
  name: string;
  department: string;
  score: number;
  salary: number;
  joinDate: string; // ISO 日期字符串
}

/** 排序方向 */
export type SortOrder = 'asc' | 'desc';

/** 字段类型 */
export type FieldType = 'number' | 'string' | 'date';

/** 排序列配置 */
export interface SortColumn<T = TableRow> {
  field: keyof T;
  order: SortOrder;
  type?: FieldType;
}

/** 表格状态 */
export interface TableState<T = TableRow> {
  data: T[];
  originalData: T[];
  sortColumns: SortColumn<T>[];
}

// ============================================================================
// 示例数据
// ============================================================================

export const SAMPLE_DATA: TableRow[] = [
  { id: 1, name: 'Alice', department: '技术部', score: 85, salary: 15000, joinDate: '2020-03-15' },
  { id: 2, name: 'Bob', department: '产品部', score: 92, salary: 18000, joinDate: '2019-07-22' },
  { id: 3, name: 'Charlie', department: '技术部', score: 85, salary: 16000, joinDate: '2021-01-10' },
  { id: 4, name: 'David', department: '技术部', score: 78, salary: 14000, joinDate: '2022-05-08' },
  { id: 5, name: 'Eve', department: '产品部', score: 92, salary: 17000, joinDate: '2020-11-30' },
  { id: 6, name: 'Frank', department: '设计部', score: 88, salary: 15500, joinDate: '2021-06-15' },
  { id: 7, name: 'Grace', department: '技术部', score: 95, salary: 20000, joinDate: '2018-09-01' },
  { id: 8, name: 'Henry', department: '产品部', score: 82, salary: 16500, joinDate: '2020-02-20' },
  { id: 9, name: 'Ivy', department: '设计部', score: 88, salary: 15000, joinDate: '2021-08-12' },
  { id: 10, name: 'Jack', department: '技术部', score: 90, salary: 18500, joinDate: '2019-12-05' },
];

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 生成更多测试数据
 */
export function generateTableData(count: number): TableRow[] {
  const departments = ['技术部', '产品部', '设计部', '运营部', '市场部'];
  const names = ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十'];

  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: `${names[i % names.length]}${Math.floor(i / names.length) + 1}`,
    department: departments[i % departments.length],
    score: 60 + Math.floor(Math.random() * 40),
    salary: 10000 + Math.floor(Math.random() * 15000),
    joinDate: new Date(
      2018 + Math.floor(Math.random() * 5),
      Math.floor(Math.random() * 12),
      1 + Math.floor(Math.random() * 28)
    ).toISOString().split('T')[0],
  }));
}

/**
 * 打印表格数据
 */
export function printTable(data: TableRow[], columns?: (keyof TableRow)[]): void {
  const cols = columns || ['id', 'name', 'department', 'score', 'salary'];

  // 表头
  console.log(cols.map(c => String(c).padEnd(12)).join('| '));
  console.log('-'.repeat(cols.length * 14));

  // 数据行
  for (const row of data) {
    console.log(cols.map(c => String(row[c]).padEnd(12)).join('| '));
  }
}

