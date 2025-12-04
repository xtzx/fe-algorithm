/**
 * ============================================================
 * 📚 富文本编辑器
 * ============================================================
 *
 * 面试考察重点：
 * 1. 编辑器架构
 * 2. 核心数据结构
 * 3. 选区与光标
 * 4. 协作编辑
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 编辑器架构类型
 *
 * 📊 L0：基于 contenteditable
 * - 依赖浏览器原生能力
 * - 简单但不可控
 * - 代表：早期 CKEditor
 *
 * 📊 L1：基于 contenteditable + 数据模型
 * - 自定义数据模型
 * - 中等复杂度
 * - 代表：Quill、Slate
 *
 * 📊 L2：完全自绘
 * - 自己实现渲染和交互
 * - 复杂但完全可控
 * - 代表：Google Docs、飞书
 *
 * 📊 主流编辑器对比
 *
 * ┌─────────────────┬────────────────────────────────────────────────┐
 * │ 编辑器          │ 特点                                           │
 * ├─────────────────┼────────────────────────────────────────────────┤
 * │ Quill           │ 简单易用，Delta 数据格式                        │
 * │ Slate           │ 框架无关，高度可定制                            │
 * │ ProseMirror     │ 底层库，功能强大，学习曲线陡                    │
 * │ TipTap          │ 基于 ProseMirror，更易用                        │
 * │ Draft.js        │ Facebook 出品，React 专用                       │
 * │ Lexical         │ Facebook 新一代，性能更好                       │
 * └─────────────────┴────────────────────────────────────────────────┘
 */

// ============================================================
// 2. 数据模型
// ============================================================

/**
 * 📊 文档数据结构
 *
 * 1. Delta（Quill）
 *    - 操作序列：insert、delete、retain
 *    - 扁平结构
 *
 * 2. 嵌套树（Slate/ProseMirror）
 *    - Document → Block → Inline → Text
 *    - 层级结构
 */

// Slate 风格的数据结构
interface SlateNode {
  type?: string;
  children?: SlateNode[];
  text?: string;
  [key: string]: any;
}

// 文档节点
interface DocumentNode extends SlateNode {
  type: 'document';
  children: BlockNode[];
}

// 块级节点
interface BlockNode extends SlateNode {
  type: 'paragraph' | 'heading' | 'list' | 'code-block' | 'image';
  children: InlineNode[];
}

// 内联节点
interface InlineNode extends SlateNode {
  type?: 'link' | 'mention';
  children: TextNode[];
}

// 文本节点
interface TextNode {
  text: string;
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
  code?: boolean;
}

// 示例文档
const documentExample: DocumentNode = {
  type: 'document',
  children: [
    {
      type: 'heading',
      level: 1,
      children: [{ text: '标题' }],
    },
    {
      type: 'paragraph',
      children: [
        { text: '这是' },
        { text: '粗体', bold: true },
        { text: '文字' },
      ],
    },
    {
      type: 'paragraph',
      children: [
        { text: '这是一个' },
        {
          type: 'link',
          url: 'https://example.com',
          children: [{ text: '链接' }],
        },
      ],
    },
  ],
};

// ============================================================
// 3. 选区与光标
// ============================================================

/**
 * 📊 选区（Selection）
 *
 * Selection API：
 * - anchor：选区起点
 * - focus：选区终点
 * - isCollapsed：是否折叠（光标状态）
 *
 * Range API：
 * - startContainer / endContainer
 * - startOffset / endOffset
 */

// 选区操作封装
class SelectionManager {
  // 获取当前选区
  static getSelection(): Selection | null {
    return window.getSelection();
  }

  // 获取选区范围
  static getRange(): Range | null {
    const selection = this.getSelection();
    if (selection && selection.rangeCount > 0) {
      return selection.getRangeAt(0);
    }
    return null;
  }

  // 设置选区
  static setRange(range: Range) {
    const selection = this.getSelection();
    if (selection) {
      selection.removeAllRanges();
      selection.addRange(range);
    }
  }

  // 创建范围
  static createRange(
    startNode: Node,
    startOffset: number,
    endNode: Node,
    endOffset: number
  ): Range {
    const range = document.createRange();
    range.setStart(startNode, startOffset);
    range.setEnd(endNode, endOffset);
    return range;
  }

  // 在指定位置插入节点
  static insertNode(node: Node) {
    const range = this.getRange();
    if (range) {
      range.deleteContents();
      range.insertNode(node);
      // 将光标移到插入节点之后
      range.setStartAfter(node);
      range.collapse(true);
      this.setRange(range);
    }
  }

  // 获取选中的文本
  static getSelectedText(): string {
    const selection = this.getSelection();
    return selection ? selection.toString() : '';
  }

  // 保存选区位置
  static saveSelection(): { anchor: PathPoint; focus: PathPoint } | null {
    const range = this.getRange();
    if (!range) return null;

    return {
      anchor: {
        node: range.startContainer,
        offset: range.startOffset,
      },
      focus: {
        node: range.endContainer,
        offset: range.endOffset,
      },
    };
  }
}

interface PathPoint {
  node: Node;
  offset: number;
}

// ============================================================
// 4. 编辑器核心操作
// ============================================================

/**
 * 📊 核心操作
 *
 * - Transform：对文档的修改操作
 * - Command：用户触发的命令
 * - Plugin：扩展功能
 */

// 操作类型
type Operation =
  | { type: 'insert_text'; path: number[]; offset: number; text: string }
  | { type: 'remove_text'; path: number[]; offset: number; text: string }
  | { type: 'insert_node'; path: number[]; node: SlateNode }
  | { type: 'remove_node'; path: number[]; node: SlateNode }
  | { type: 'set_node'; path: number[]; properties: Partial<SlateNode> };

// 编辑器核心类
class Editor {
  document: DocumentNode;
  selection: { anchor: number[]; focus: number[] } | null = null;
  history: { undos: Operation[][]; redos: Operation[][] } = { undos: [], redos: [] };
  private listeners: Map<string, Function[]> = new Map();

  constructor(initialDocument: DocumentNode) {
    this.document = initialDocument;
  }

  // 应用操作
  apply(operation: Operation) {
    switch (operation.type) {
      case 'insert_text':
        this.insertText(operation.path, operation.offset, operation.text);
        break;
      case 'remove_text':
        this.removeText(operation.path, operation.offset, operation.text.length);
        break;
      // ... 其他操作
    }

    this.emit('change', { operation });
  }

  // 插入文本
  private insertText(path: number[], offset: number, text: string) {
    const node = this.getNode(path) as TextNode;
    if (node && 'text' in node) {
      node.text = node.text.slice(0, offset) + text + node.text.slice(offset);
    }
  }

  // 删除文本
  private removeText(path: number[], offset: number, length: number) {
    const node = this.getNode(path) as TextNode;
    if (node && 'text' in node) {
      node.text = node.text.slice(0, offset) + node.text.slice(offset + length);
    }
  }

  // 获取节点
  private getNode(path: number[]): SlateNode | null {
    let node: SlateNode = this.document;
    for (const index of path) {
      if (node.children && node.children[index]) {
        node = node.children[index];
      } else {
        return null;
      }
    }
    return node;
  }

  // 事件监听
  on(event: string, handler: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(handler);
  }

  private emit(event: string, data: any) {
    const handlers = this.listeners.get(event) || [];
    handlers.forEach(handler => handler(data));
  }

  // 撤销
  undo() {
    const operations = this.history.undos.pop();
    if (operations) {
      // 反向应用操作
      this.history.redos.push(operations);
    }
  }

  // 重做
  redo() {
    const operations = this.history.redos.pop();
    if (operations) {
      operations.forEach(op => this.apply(op));
      this.history.undos.push(operations);
    }
  }
}

// ============================================================
// 5. 协作编辑（CRDT/OT）
// ============================================================

/**
 * 📊 协作编辑算法
 *
 * OT（Operational Transformation）：
 * - 服务端协调
 * - 操作转换
 * - Google Docs 使用
 *
 * CRDT（Conflict-free Replicated Data Types）：
 * - 无需服务端协调
 * - 最终一致性
 * - Yjs、Automerge 使用
 */

// CRDT 概念示例（Yjs 风格）
const crdtExample = `
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';

// 创建 Yjs 文档
const ydoc = new Y.Doc();

// 获取共享类型
const ytext = ydoc.getText('content');

// 连接 WebSocket
const provider = new WebsocketProvider(
  'wss://your-server.com',
  'room-name',
  ydoc
);

// 监听变化
ytext.observe(event => {
  console.log('Text changed:', ytext.toString());
});

// 编辑
ytext.insert(0, 'Hello ');
ytext.insert(6, 'World');

// 与 Slate 集成
import { withYjs, slateNodesToInsertDelta } from '@slate-yjs/core';

const editor = withYjs(createEditor(), sharedType);
`;

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. 浏览器兼容性
 *    - contenteditable 行为不一致
 *    - 需要大量兼容处理
 *
 * 2. 选区丢失
 *    - 失去焦点时选区消失
 *    - 需要保存/恢复选区
 *
 * 3. 输入法问题
 *    - 中文输入需要 compositionstart/end
 *    - 避免在输入过程中修改 DOM
 *
 * 4. 性能问题
 *    - 大文档渲染慢
 *    - 使用虚拟滚动
 *
 * 5. 协作冲突
 *    - 同时编辑同一位置
 *    - 使用 CRDT/OT 解决
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 如何设计一个富文本编辑器？
 * A:
 *    - 数据模型（嵌套树或扁平序列）
 *    - 选区管理
 *    - 操作系统（Transform）
 *    - 撤销/重做
 *    - 插件系统
 *
 * Q2: OT 和 CRDT 的区别？
 * A:
 *    OT：
 *    - 需要服务端协调
 *    - 操作转换保证一致性
 *
 *    CRDT：
 *    - 无需中心服务器
 *    - 数据结构本身保证一致性
 *
 * Q3: 如何处理中文输入？
 * A:
 *    - 监听 compositionstart/compositionend
 *    - 输入过程中不修改 DOM
 *    - 输入完成后再更新
 *
 * Q4: 如何优化大文档性能？
 * A:
 *    - 虚拟滚动
 *    - 分块渲染
 *    - 延迟渲染不可见部分
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：Markdown 编辑器
 */

const markdownEditorExample = `
// Markdown 编辑器架构
┌─────────────────────────────────────────────────────────────────┐
│                      Markdown 编辑器                             │
│                                                                 │
│  ┌────────────────────────┐  ┌────────────────────────┐        │
│  │      编辑区域          │  │      预览区域          │        │
│  │                        │  │                        │        │
│  │  Markdown 输入         │  │  HTML 渲染            │        │
│  │                        │  │                        │        │
│  │  实时同步 ─────────────│──│► 实时预览            │        │
│  │                        │  │                        │        │
│  └────────────────────────┘  └────────────────────────┘        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                        工具栏                             │  │
│  │  标题 │ 粗体 │ 斜体 │ 链接 │ 图片 │ 代码 │ 列表          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

// 核心逻辑
class MarkdownEditor {
  private input: HTMLTextAreaElement;
  private preview: HTMLDivElement;
  private parser: MarkdownParser;

  constructor(container: HTMLElement) {
    this.input = container.querySelector('.editor')!;
    this.preview = container.querySelector('.preview')!;
    this.parser = new MarkdownParser();
    
    this.input.addEventListener('input', this.handleInput);
  }

  private handleInput = debounce(() => {
    const markdown = this.input.value;
    const html = this.parser.parse(markdown);
    this.preview.innerHTML = html;
  }, 100);

  insertText(text: string) {
    const { selectionStart, selectionEnd } = this.input;
    const before = this.input.value.slice(0, selectionStart);
    const after = this.input.value.slice(selectionEnd);
    this.input.value = before + text + after;
    this.input.selectionStart = selectionStart + text.length;
    this.input.selectionEnd = selectionStart + text.length;
    this.handleInput();
  }
}
`;

// 模拟函数
function debounce(fn: Function, delay: number) {
  let timer: ReturnType<typeof setTimeout>;
  return function(...args: any[]) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

export {
  SelectionManager,
  Editor,
  documentExample,
  crdtExample,
  markdownEditorExample,
};

