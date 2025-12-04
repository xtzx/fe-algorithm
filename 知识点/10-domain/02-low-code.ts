/**
 * ============================================================
 * 📚 低代码平台
 * ============================================================
 *
 * 面试考察重点：
 * 1. 低代码核心概念
 * 2. Schema 设计
 * 3. 拖拽实现
 * 4. 渲染引擎
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是低代码？
 *
 * 通过可视化方式，用少量代码或无代码构建应用。
 *
 * 📊 低代码 vs 无代码
 *
 * 低代码（Low-Code）：
 * - 面向开发者
 * - 支持代码扩展
 * - 灵活度高
 *
 * 无代码（No-Code）：
 * - 面向业务人员
 * - 纯可视化配置
 * - 灵活度受限
 *
 * 📊 低代码平台核心模块
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        低代码平台架构                            │
 * │                                                                 │
 * │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
 * │  │  设计器     │  │  渲染器     │  │  物料库     │             │
 * │  │  Designer   │  │  Renderer   │  │  Materials  │             │
 * │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
 * │         │                │                │                     │
 * │         └────────────────┼────────────────┘                     │
 * │                          │                                      │
 * │                          ▼                                      │
 * │                   ┌─────────────┐                               │
 * │                   │   Schema    │                               │
 * │                   │   (JSON)    │                               │
 * │                   └─────────────┘                               │
 * │                                                                 │
 * │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
 * │  │  出码引擎   │  │  数据源     │  │  逻辑编排   │             │
 * │  │  Code Gen   │  │  DataSource │  │  Logic Flow │             │
 * │  └─────────────┘  └─────────────┘  └─────────────┘             │
 * └─────────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 2. Schema 设计
// ============================================================

/**
 * 📊 Schema 规范
 *
 * 核心字段：
 * - componentName：组件名
 * - props：组件属性
 * - children：子组件
 * - id：唯一标识
 */

// 组件 Schema 定义
interface ComponentSchema {
  id: string;
  componentName: string;
  props: Record<string, any>;
  children?: ComponentSchema[];
  // 样式
  style?: React.CSSProperties;
  // 事件
  events?: Record<string, EventSchema>;
  // 循环渲染
  loop?: {
    data: string; // 数据源表达式
    item: string; // 循环变量名
  };
  // 条件渲染
  condition?: string; // 表达式
}

interface EventSchema {
  type: 'action' | 'script';
  // action 类型
  action?: string;
  params?: Record<string, any>;
  // script 类型
  script?: string;
}

// 页面 Schema
interface PageSchema {
  version: string;
  componentsTree: ComponentSchema[];
  state: Record<string, any>;
  methods: Record<string, string>;
  dataSource: DataSourceSchema[];
  lifeCycles: LifeCycleSchema;
}

interface DataSourceSchema {
  id: string;
  type: 'api' | 'static';
  options: {
    url?: string;
    method?: string;
    params?: Record<string, any>;
    data?: any;
  };
}

interface LifeCycleSchema {
  onMount?: string;
  onUnmount?: string;
}

// Schema 示例
const schemaExample = `
{
  "version": "1.0.0",
  "componentsTree": [
    {
      "id": "container-1",
      "componentName": "Container",
      "props": { "className": "page-container" },
      "children": [
        {
          "id": "form-1",
          "componentName": "Form",
          "props": { "labelCol": { "span": 4 } },
          "children": [
            {
              "id": "input-1",
              "componentName": "Input",
              "props": {
                "label": "用户名",
                "name": "username",
                "placeholder": "请输入用户名"
              }
            },
            {
              "id": "button-1",
              "componentName": "Button",
              "props": {
                "type": "primary",
                "children": "提交"
              },
              "events": {
                "onClick": {
                  "type": "action",
                  "action": "submitForm"
                }
              }
            }
          ]
        }
      ]
    }
  ],
  "state": {
    "formData": {}
  },
  "methods": {
    "submitForm": "async function() { await this.dataSource.api1.fetch(); }"
  }
}
`;

// ============================================================
// 3. 拖拽实现
// ============================================================

/**
 * 📊 拖拽核心
 *
 * HTML5 Drag and Drop API：
 * - dragstart：开始拖拽
 * - drag：拖拽中
 * - dragend：拖拽结束
 * - dragenter：进入目标
 * - dragover：在目标上移动
 * - dragleave：离开目标
 * - drop：放置
 */

// 拖拽管理器
class DragDropManager {
  private draggingData: any = null;
  private dropTargets: Map<HTMLElement, DropHandler> = new Map();

  // 设置可拖拽元素
  makeDraggable(element: HTMLElement, data: any) {
    element.draggable = true;

    element.addEventListener('dragstart', (e) => {
      this.draggingData = data;
      e.dataTransfer!.effectAllowed = 'move';
      element.classList.add('dragging');
    });

    element.addEventListener('dragend', () => {
      this.draggingData = null;
      element.classList.remove('dragging');
    });
  }

  // 设置放置目标
  makeDroppable(element: HTMLElement, handler: DropHandler) {
    this.dropTargets.set(element, handler);

    element.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer!.dropEffect = 'move';
      element.classList.add('drag-over');
    });

    element.addEventListener('dragleave', () => {
      element.classList.remove('drag-over');
    });

    element.addEventListener('drop', (e) => {
      e.preventDefault();
      element.classList.remove('drag-over');

      if (this.draggingData) {
        const position = this.calculateDropPosition(e, element);
        handler.onDrop(this.draggingData, position);
      }
    });
  }

  // 计算放置位置
  private calculateDropPosition(e: DragEvent, target: HTMLElement): DropPosition {
    const rect = target.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const threshold = rect.height / 3;

    if (y < threshold) return 'before';
    if (y > rect.height - threshold) return 'after';
    return 'inside';
  }
}

type DropPosition = 'before' | 'after' | 'inside';
interface DropHandler {
  onDrop: (data: any, position: DropPosition) => void;
}

// ============================================================
// 4. 渲染引擎
// ============================================================

/**
 * 📊 渲染引擎核心
 *
 * 1. 组件映射：componentName → Component
 * 2. 属性解析：处理表达式、数据绑定
 * 3. 事件绑定：绑定 events
 * 4. 递归渲染：处理 children
 */

// 组件注册表
const componentRegistry: Map<string, React.ComponentType<any>> = new Map();

function registerComponent(name: string, component: React.ComponentType<any>) {
  componentRegistry.set(name, component);
}

// 渲染引擎（简化版）
const schemaRendererCode = `
import React from 'react';

interface RendererProps {
  schema: ComponentSchema;
  context: RendererContext;
}

interface RendererContext {
  state: Record<string, any>;
  setState: (key: string, value: any) => void;
  methods: Record<string, Function>;
  dataSource: Record<string, any>;
}

function SchemaRenderer({ schema, context }: RendererProps) {
  const { componentName, props, children, events, loop, condition, style } = schema;

  // 1. 条件渲染
  if (condition) {
    const result = evaluateExpression(condition, context);
    if (!result) return null;
  }

  // 2. 获取组件
  const Component = componentRegistry.get(componentName);
  if (!Component) {
    console.warn(\`Component not found: \${componentName}\`);
    return null;
  }

  // 3. 解析 props（处理表达式）
  const resolvedProps = resolveProps(props, context);

  // 4. 绑定事件
  const eventHandlers = bindEvents(events, context);

  // 5. 循环渲染
  if (loop) {
    const dataSource = evaluateExpression(loop.data, context);
    return dataSource.map((item: any, index: number) => {
      const loopContext = {
        ...context,
        [loop.item]: item,
        index,
      };
      return (
        <Component
          key={index}
          {...resolvedProps}
          {...eventHandlers}
          style={style}
        >
          {children?.map(child => (
            <SchemaRenderer
              key={child.id}
              schema={child}
              context={loopContext}
            />
          ))}
        </Component>
      );
    });
  }

  // 6. 普通渲染
  return (
    <Component {...resolvedProps} {...eventHandlers} style={style}>
      {children?.map(child => (
        <SchemaRenderer key={child.id} schema={child} context={context} />
      ))}
    </Component>
  );
}

// 解析 props 中的表达式
function resolveProps(props: Record<string, any>, context: RendererContext) {
  const resolved: Record<string, any> = {};

  for (const [key, value] of Object.entries(props)) {
    if (typeof value === 'string' && value.startsWith('{{') && value.endsWith('}}')) {
      // 表达式
      const expression = value.slice(2, -2).trim();
      resolved[key] = evaluateExpression(expression, context);
    } else {
      resolved[key] = value;
    }
  }

  return resolved;
}

// 执行表达式
function evaluateExpression(expression: string, context: RendererContext) {
  const { state, methods, dataSource } = context;
  try {
    // 使用 Function 构造器执行表达式
    return new Function('state', 'methods', 'dataSource', \`return \${expression}\`)(
      state,
      methods,
      dataSource
    );
  } catch (e) {
    console.error('Expression error:', expression, e);
    return undefined;
  }
}
`;

// ============================================================
// 5. 物料系统
// ============================================================

/**
 * 📊 物料定义
 *
 * 物料 = 组件 + 配置面板
 */

interface MaterialConfig {
  name: string;
  title: string;
  category: string;
  icon: string;
  // 组件
  component: React.ComponentType<any>;
  // 默认 props
  defaultProps: Record<string, any>;
  // 配置面板
  configure: PropertyConfig[];
}

interface PropertyConfig {
  name: string;
  title: string;
  type: 'string' | 'number' | 'boolean' | 'select' | 'json' | 'expression';
  default?: any;
  options?: { label: string; value: any }[];
}

// 物料示例
const buttonMaterial: MaterialConfig = {
  name: 'Button',
  title: '按钮',
  category: '基础组件',
  icon: 'button-icon',
  component: () => null, // 实际组件
  defaultProps: {
    type: 'default',
    children: '按钮',
  },
  configure: [
    {
      name: 'type',
      title: '类型',
      type: 'select',
      options: [
        { label: '默认', value: 'default' },
        { label: '主要', value: 'primary' },
        { label: '危险', value: 'danger' },
      ],
    },
    {
      name: 'children',
      title: '文本',
      type: 'string',
    },
    {
      name: 'disabled',
      title: '禁用',
      type: 'boolean',
      default: false,
    },
  ],
};

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. Schema 设计不合理
 *    - 扩展性差
 *    - 建议参考 AliLowCodeEngine 规范
 *
 * 2. 表达式安全问题
 *    - eval 有安全风险
 *    - 使用沙箱执行
 *
 * 3. 性能问题
 *    - 大量组件卡顿
 *    - 使用虚拟化
 *
 * 4. 拖拽体验差
 *    - 需要吸附、辅助线
 *    - 撤销/重做功能
 *
 * 5. 出码质量差
 *    - 生成的代码不可维护
 *    - 优化代码生成逻辑
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 低代码的核心模块有哪些？
 * A:
 *    - 设计器（拖拽、配置）
 *    - 渲染器（Schema → 页面）
 *    - 物料系统（组件库）
 *    - 出码引擎（Schema → 代码）
 *    - 数据源管理
 *    - 逻辑编排
 *
 * Q2: 如何设计 Schema？
 * A:
 *    - 组件树结构
 *    - props 支持表达式
 *    - 事件定义
 *    - 循环/条件渲染
 *    - 可扩展
 *
 * Q3: 如何实现表达式求值？
 * A:
 *    - new Function 或 eval
 *    - 沙箱隔离
 *    - 错误处理
 *
 * Q4: 低代码的局限性？
 * A:
 *    - 复杂逻辑难以实现
 *    - 定制化程度有限
 *    - 出码质量参差
 *    - 调试困难
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：表单设计器
 */

const formDesignerExample = `
// 表单设计器架构
┌─────────────────────────────────────────────────────────────────┐
│                         表单设计器                               │
│                                                                 │
│  ┌────────────┐  ┌────────────────────────┐  ┌────────────┐    │
│  │  物料面板   │  │      画布区域          │  │  配置面板   │    │
│  │            │  │                        │  │            │    │
│  │  输入框     │  │  ┌────────────────┐   │  │  属性配置   │    │
│  │  选择器     │  │  │                │   │  │            │    │
│  │  日期       │  │  │  拖入组件      │   │  │  校验规则   │    │
│  │  上传       │  │  │                │   │  │            │    │
│  │  ...       │  │  └────────────────┘   │  │  联动配置   │    │
│  │            │  │                        │  │            │    │
│  └────────────┘  └────────────────────────┘  └────────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      工具栏                               │  │
│  │  预览 │ 保存 │ 发布 │ 撤销 │ 重做 │ 清空                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

// 核心状态
interface DesignerState {
  schema: FormSchema;
  selectedId: string | null;
  history: FormSchema[];
  historyIndex: number;
}
`;

export {
  DragDropManager,
  registerComponent,
  componentRegistry,
  buttonMaterial,
  schemaExample,
  schemaRendererCode,
  formDesignerExample,
};

