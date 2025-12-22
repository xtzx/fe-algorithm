/**
 * ============================================================
 * 📚 二叉树 - 前端业务场景代码示例
 * ============================================================
 *
 * 本文件展示二叉树在前端实际业务中的应用
 */

// ============================================================
// 基础节点定义
// ============================================================

interface TreeNode<T = unknown> {
  value: T;
  children: TreeNode<T>[];
}

// ============================================================
// 1. 虚拟 DOM Diff 算法简化版
// ============================================================

/**
 * 📝 业务场景：React/Vue 虚拟 DOM
 *
 * 场景描述：
 * - 比较新旧虚拟 DOM 树
 * - 生成最小更新操作
 */
interface VNode {
  type: string;
  props: Record<string, unknown>;
  children: VNode[];
  key?: string | number;
}

type PatchType = 'CREATE' | 'REMOVE' | 'REPLACE' | 'UPDATE';

interface Patch {
  type: PatchType;
  node?: VNode;
  props?: Record<string, unknown>;
}

function diff(oldNode: VNode | null, newNode: VNode | null): Patch | null {
  // 新节点不存在，删除
  if (!newNode) {
    return { type: 'REMOVE' };
  }

  // 旧节点不存在，创建
  if (!oldNode) {
    return { type: 'CREATE', node: newNode };
  }

  // 类型不同，替换
  if (oldNode.type !== newNode.type) {
    return { type: 'REPLACE', node: newNode };
  }

  // 类型相同，比较属性
  const propsPatches = diffProps(oldNode.props, newNode.props);

  if (Object.keys(propsPatches).length > 0) {
    return { type: 'UPDATE', props: propsPatches };
  }

  return null;
}

function diffProps(
  oldProps: Record<string, unknown>,
  newProps: Record<string, unknown>
): Record<string, unknown> {
  const patches: Record<string, unknown> = {};

  // 检查新增和修改的属性
  for (const key in newProps) {
    if (oldProps[key] !== newProps[key]) {
      patches[key] = newProps[key];
    }
  }

  // 检查删除的属性
  for (const key in oldProps) {
    if (!(key in newProps)) {
      patches[key] = null;
    }
  }

  return patches;
}

/**
 * 递归 Diff 子节点
 */
function diffChildren(
  oldChildren: VNode[],
  newChildren: VNode[]
): (Patch | null)[] {
  const patches: (Patch | null)[] = [];
  const maxLen = Math.max(oldChildren.length, newChildren.length);

  for (let i = 0; i < maxLen; i++) {
    patches.push(diff(oldChildren[i] || null, newChildren[i] || null));
  }

  return patches;
}

// ============================================================
// 2. 组件树遍历与生命周期
// ============================================================

/**
 * 📝 业务场景：React 组件树
 *
 * 场景描述：
 * - 模拟组件的挂载和卸载顺序
 * - 父组件先挂载，子组件后挂载（前序）
 * - 子组件先卸载，父组件后卸载（后序）
 */
interface Component {
  name: string;
  children: Component[];
  mounted?: boolean;
}

class ComponentLifecycle {
  /**
   * 挂载组件树（前序遍历）
   * 父组件的 componentWillMount 在子组件之前
   */
  mount(component: Component): void {
    // 前序位置：挂载当前组件
    console.log(`Mounting: ${component.name}`);
    component.mounted = true;

    // 递归挂载子组件
    for (const child of component.children) {
      this.mount(child);
    }

    // 后序位置：componentDidMount
    console.log(`Mounted: ${component.name}`);
  }

  /**
   * 卸载组件树（后序遍历）
   * 子组件先卸载，父组件后卸载
   */
  unmount(component: Component): void {
    // 先递归卸载子组件
    for (const child of component.children) {
      this.unmount(child);
    }

    // 后序位置：卸载当前组件
    console.log(`Unmounting: ${component.name}`);
    component.mounted = false;
  }

  /**
   * 收集所有已挂载的组件（层序遍历）
   */
  getAllMounted(root: Component): string[] {
    const result: string[] = [];
    const queue: Component[] = [root];

    while (queue.length > 0) {
      const component = queue.shift()!;
      if (component.mounted) {
        result.push(component.name);
      }
      queue.push(...component.children);
    }

    return result;
  }
}

// ============================================================
// 3. 菜单/导航树组件
// ============================================================

/**
 * 📝 业务场景：侧边栏菜单
 *
 * 场景描述：
 * - 多层级菜单
 * - 支持展开/收起
 * - 根据路由高亮当前项
 */
interface MenuItem {
  id: string;
  label: string;
  path?: string;
  icon?: string;
  children?: MenuItem[];
  expanded?: boolean;
}

class MenuTree {
  /**
   * 根据路径查找菜单项
   */
  findByPath(menu: MenuItem[], path: string): MenuItem | null {
    for (const item of menu) {
      if (item.path === path) {
        return item;
      }
      if (item.children) {
        const found = this.findByPath(item.children, path);
        if (found) return found;
      }
    }
    return null;
  }

  /**
   * 获取菜单项的路径（面包屑）
   */
  getBreadcrumb(menu: MenuItem[], targetId: string): MenuItem[] {
    const path: MenuItem[] = [];

    const dfs = (items: MenuItem[]): boolean => {
      for (const item of items) {
        path.push(item);

        if (item.id === targetId) {
          return true;
        }

        if (item.children && dfs(item.children)) {
          return true;
        }

        path.pop(); // 回溯
      }
      return false;
    };

    dfs(menu);
    return path;
  }

  /**
   * 展开到指定节点
   */
  expandToNode(menu: MenuItem[], targetId: string): void {
    const dfs = (items: MenuItem[]): boolean => {
      for (const item of items) {
        if (item.id === targetId) {
          return true;
        }

        if (item.children && dfs(item.children)) {
          item.expanded = true;
          return true;
        }
      }
      return false;
    };

    dfs(menu);
  }

  /**
   * 扁平化菜单（用于搜索）
   */
  flatten(menu: MenuItem[]): MenuItem[] {
    const result: MenuItem[] = [];

    const dfs = (items: MenuItem[]) => {
      for (const item of items) {
        result.push(item);
        if (item.children) {
          dfs(item.children);
        }
      }
    };

    dfs(menu);
    return result;
  }
}

// ============================================================
// 4. 文件系统树
// ============================================================

/**
 * 📝 业务场景：文件管理器
 *
 * 场景描述：
 * - 展示文件夹结构
 * - 计算文件夹大小
 * - 搜索文件
 */
interface FileNode {
  name: string;
  type: 'file' | 'folder';
  size?: number; // 文件大小（字节）
  children?: FileNode[];
}

class FileSystemTree {
  /**
   * 计算文件夹总大小（后序遍历）
   */
  calculateSize(node: FileNode): number {
    if (node.type === 'file') {
      return node.size || 0;
    }

    let totalSize = 0;
    for (const child of node.children || []) {
      totalSize += this.calculateSize(child);
    }

    return totalSize;
  }

  /**
   * 搜索文件（DFS）
   */
  search(root: FileNode, keyword: string): FileNode[] {
    const results: FileNode[] = [];

    const dfs = (node: FileNode, path: string) => {
      const currentPath = path ? `${path}/${node.name}` : node.name;

      if (node.name.toLowerCase().includes(keyword.toLowerCase())) {
        results.push({ ...node, name: currentPath });
      }

      if (node.children) {
        for (const child of node.children) {
          dfs(child, currentPath);
        }
      }
    };

    dfs(root, '');
    return results;
  }

  /**
   * 获取目录结构字符串
   */
  printTree(node: FileNode, prefix = '', isLast = true): string {
    let result =
      prefix + (isLast ? '└── ' : '├── ') + node.name + '\n';

    if (node.children) {
      const childPrefix = prefix + (isLast ? '    ' : '│   ');
      node.children.forEach((child, index) => {
        const isChildLast = index === node.children!.length - 1;
        result += this.printTree(child, childPrefix, isChildLast);
      });
    }

    return result;
  }
}

// ============================================================
// 5. 评论树（无限嵌套回复）
// ============================================================

/**
 * 📝 业务场景：评论回复系统
 *
 * 场景描述：
 * - 评论可以无限嵌套回复
 * - 支持折叠/展开
 * - 统计回复数量
 */
interface Comment {
  id: string;
  content: string;
  author: string;
  createdAt: Date;
  replies: Comment[];
  collapsed?: boolean;
}

class CommentTree {
  /**
   * 统计总回复数（后序遍历）
   */
  countReplies(comment: Comment): number {
    let count = 0;
    for (const reply of comment.replies) {
      count += 1 + this.countReplies(reply);
    }
    return count;
  }

  /**
   * 找到指定评论
   */
  findComment(root: Comment, targetId: string): Comment | null {
    if (root.id === targetId) {
      return root;
    }

    for (const reply of root.replies) {
      const found = this.findComment(reply, targetId);
      if (found) return found;
    }

    return null;
  }

  /**
   * 添加回复
   */
  addReply(root: Comment, parentId: string, newReply: Comment): boolean {
    const parent = this.findComment(root, parentId);
    if (parent) {
      parent.replies.push(newReply);
      return true;
    }
    return false;
  }

  /**
   * 删除评论（及其所有回复）
   */
  deleteComment(root: Comment, targetId: string): boolean {
    for (let i = 0; i < root.replies.length; i++) {
      if (root.replies[i].id === targetId) {
        root.replies.splice(i, 1);
        return true;
      }
      if (this.deleteComment(root.replies[i], targetId)) {
        return true;
      }
    }
    return false;
  }

  /**
   * 获取评论链（从根到目标）
   */
  getCommentChain(root: Comment, targetId: string): Comment[] {
    const chain: Comment[] = [];

    const dfs = (comment: Comment): boolean => {
      chain.push(comment);

      if (comment.id === targetId) {
        return true;
      }

      for (const reply of comment.replies) {
        if (dfs(reply)) {
          return true;
        }
      }

      chain.pop();
      return false;
    };

    dfs(root);
    return chain;
  }
}

// ============================================================
// 6. AST 遍历与转换
// ============================================================

/**
 * 📝 业务场景：代码转换工具
 *
 * 场景描述：
 * - 遍历 AST 节点
 * - 修改特定节点
 * - 类似 Babel 插件
 */
interface ASTNode {
  type: string;
  value?: string | number;
  children?: ASTNode[];
  [key: string]: unknown;
}

type Visitor = {
  [key: string]: (node: ASTNode, parent?: ASTNode) => void;
};

function traverseAST(
  node: ASTNode,
  visitor: Visitor,
  parent?: ASTNode
): void {
  // 调用对应类型的 visitor
  const handler = visitor[node.type];
  if (handler) {
    handler(node, parent);
  }

  // 递归遍历子节点
  if (node.children) {
    for (const child of node.children) {
      traverseAST(child, visitor, node);
    }
  }
}

// 使用示例：将所有变量名转为大写
const uppercaseVisitor: Visitor = {
  Identifier: (node) => {
    if (typeof node.value === 'string') {
      node.value = node.value.toUpperCase();
    }
  },
};

// ============================================================
// 7. 组织架构树
// ============================================================

/**
 * 📝 业务场景：人员组织架构
 *
 * 场景描述：
 * - 展示公司组织架构
 * - 查找上下级关系
 * - 统计部门人数
 */
interface OrgNode {
  id: string;
  name: string;
  title: string;
  department: string;
  subordinates: OrgNode[];
}

class OrgTree {
  /**
   * 找到某人的所有上级
   */
  findSuperiors(root: OrgNode, targetId: string): OrgNode[] {
    const superiors: OrgNode[] = [];

    const dfs = (node: OrgNode): boolean => {
      if (node.id === targetId) {
        return true;
      }

      for (const sub of node.subordinates) {
        if (dfs(sub)) {
          superiors.unshift(node);
          return true;
        }
      }

      return false;
    };

    dfs(root);
    return superiors;
  }

  /**
   * 统计某人管理的总人数
   */
  countSubordinates(node: OrgNode): number {
    let count = 0;
    for (const sub of node.subordinates) {
      count += 1 + this.countSubordinates(sub);
    }
    return count;
  }

  /**
   * 按部门分组
   */
  groupByDepartment(root: OrgNode): Map<string, OrgNode[]> {
    const groups = new Map<string, OrgNode[]>();

    const dfs = (node: OrgNode) => {
      const list = groups.get(node.department) || [];
      list.push(node);
      groups.set(node.department, list);

      for (const sub of node.subordinates) {
        dfs(sub);
      }
    };

    dfs(root);
    return groups;
  }

  /**
   * 找到两个人的最近公共上级
   */
  findCommonSuperior(
    root: OrgNode,
    id1: string,
    id2: string
  ): OrgNode | null {
    if (root.id === id1 || root.id === id2) {
      return root;
    }

    let foundIn: OrgNode | null = null;
    let count = 0;

    for (const sub of root.subordinates) {
      const result = this.findCommonSuperior(sub, id1, id2);
      if (result) {
        foundIn = result;
        count++;
      }
    }

    if (count === 2) {
      return root; // 分别在不同子树中找到
    }

    return foundIn;
  }
}

// ============================================================
// 导出
// ============================================================

export {
  VNode,
  diff,
  diffProps,
  diffChildren,
  ComponentLifecycle,
  MenuTree,
  FileSystemTree,
  CommentTree,
  traverseAST,
  OrgTree,
};

export type {
  TreeNode,
  Component,
  MenuItem,
  FileNode,
  Comment,
  ASTNode,
  Visitor,
  OrgNode,
};

