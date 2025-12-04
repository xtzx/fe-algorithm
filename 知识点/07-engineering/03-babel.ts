/**
 * ============================================================
 * 📚 Babel 与 AST
 * ============================================================
 *
 * 面试考察重点：
 * 1. Babel 的作用和工作原理
 * 2. AST 的概念和应用
 * 3. Babel 插件开发
 * 4. Polyfill 策略
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是 Babel？
 *
 * Babel 是 JavaScript 编译器，主要功能：
 * 1. 语法转换：ES6+ → ES5
 * 2. Polyfill：API 垫片
 * 3. 源码转换：JSX、TypeScript
 *
 * 📊 Babel 工作流程
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                                                                 │
 * │   源代码 ──► Parse ──► AST ──► Transform ──► AST ──► Generate   │
 * │              解析          抽象语法树   转换                生成  │
 * │                               │                                │
 * │                          Plugin 在这里处理                      │
 * │                                                                 │
 * └─────────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 2. AST（抽象语法树）
// ============================================================

/**
 * 📊 什么是 AST？
 *
 * AST 是源代码的树状结构表示。
 *
 * 代码：const a = 1 + 2;
 *
 * AST：
 * {
 *   type: "VariableDeclaration",
 *   kind: "const",
 *   declarations: [{
 *     type: "VariableDeclarator",
 *     id: { type: "Identifier", name: "a" },
 *     init: {
 *       type: "BinaryExpression",
 *       operator: "+",
 *       left: { type: "NumericLiteral", value: 1 },
 *       right: { type: "NumericLiteral", value: 2 }
 *     }
 *   }]
 * }
 */

/**
 * 📊 常见 AST 节点类型
 *
 * - Program：程序根节点
 * - VariableDeclaration：变量声明
 * - FunctionDeclaration：函数声明
 * - Identifier：标识符
 * - Literal：字面量
 * - BinaryExpression：二元表达式
 * - CallExpression：函数调用
 * - MemberExpression：成员表达式
 * - ArrowFunctionExpression：箭头函数
 * - ImportDeclaration：导入声明
 * - ExportDeclaration：导出声明
 */

// ============================================================
// 3. Babel 配置
// ============================================================

const babelConfigExample = `
// babel.config.js
module.exports = {
  presets: [
    [
      '@babel/preset-env',
      {
        // 目标环境
        targets: {
          browsers: ['> 1%', 'last 2 versions', 'not dead'],
          node: 'current',
        },
        // 按需引入 polyfill
        useBuiltIns: 'usage',
        corejs: 3,
        // 使用 ES modules
        modules: false,
      },
    ],
    '@babel/preset-react',
    '@babel/preset-typescript',
  ],
  plugins: [
    '@babel/plugin-proposal-decorators',
    '@babel/plugin-transform-runtime',
  ],
};
`;

/**
 * 📊 preset vs plugin
 *
 * Plugin：单个转换功能
 * Preset：一组 Plugin 的集合
 *
 * 执行顺序：
 * - Plugin 先执行，从前到后
 * - Preset 后执行，从后到前
 */

/**
 * 📊 @babel/preset-env
 *
 * 智能预设，根据目标环境自动确定需要的转换：
 *
 * targets：目标环境
 * - browsers：浏览器列表
 * - node：Node.js 版本
 *
 * useBuiltIns：polyfill 策略
 * - false：不引入 polyfill
 * - entry：入口处全量引入
 * - usage：按使用自动引入（推荐）
 *
 * corejs：core-js 版本
 */

// ============================================================
// 4. Polyfill 策略
// ============================================================

/**
 * 📊 Polyfill vs 语法转换
 *
 * 语法转换：箭头函数、解构、class 等
 * - Babel 可以直接转换
 *
 * API Polyfill：Promise、Array.includes 等
 * - 需要额外引入
 * - core-js 提供
 *
 * 📊 Polyfill 引入方式
 *
 * 1. useBuiltIns: 'entry'
 *    - 入口处 import 'core-js'
 *    - 全量引入，体积大
 *
 * 2. useBuiltIns: 'usage'
 *    - 按使用自动引入
 *    - 推荐
 *
 * 3. @babel/plugin-transform-runtime
 *    - 复用 helper 函数
 *    - 避免全局污染
 *    - 适合库开发
 */

const runtimePluginExample = `
// 不使用 @babel/plugin-transform-runtime
// 每个文件都会内联 helper 函数
function _classCallCheck(instance, Constructor) { ... }
function _defineProperties(target, props) { ... }

// 使用后，从 @babel/runtime 导入
import _classCallCheck from "@babel/runtime/helpers/classCallCheck";
import _defineProperties from "@babel/runtime/helpers/defineProperties";

// 配置
{
  "plugins": [
    ["@babel/plugin-transform-runtime", {
      "corejs": 3,  // 使用 @babel/runtime-corejs3
      "helpers": true,
      "regenerator": true
    }]
  ]
}
`;

// ============================================================
// 5. Babel 插件开发
// ============================================================

/**
 * 📊 Babel 插件结构
 *
 * 插件是一个函数，返回一个包含 visitor 的对象。
 * visitor 定义了如何处理各种 AST 节点。
 */

// 简单的 console.log 移除插件
const removeConsolePlugin = `
module.exports = function() {
  return {
    name: 'remove-console',
    visitor: {
      CallExpression(path) {
        const callee = path.node.callee;
        
        // 检查是否是 console.xxx
        if (
          callee.type === 'MemberExpression' &&
          callee.object.name === 'console'
        ) {
          // 移除这个节点
          path.remove();
        }
      },
    },
  };
};
`;

// 自动添加 try-catch 的插件
const autoTryCatchPlugin = `
const t = require('@babel/types');

module.exports = function() {
  return {
    name: 'auto-try-catch',
    visitor: {
      // 处理 async 函数
      'FunctionDeclaration|ArrowFunctionExpression|FunctionExpression'(path) {
        if (!path.node.async) return;
        
        const body = path.node.body;
        if (body.type !== 'BlockStatement') return;
        
        // 已经有 try-catch 的跳过
        if (
          body.body.length === 1 &&
          body.body[0].type === 'TryStatement'
        ) {
          return;
        }
        
        // 包装成 try-catch
        const tryStatement = t.tryStatement(
          t.blockStatement(body.body),
          t.catchClause(
            t.identifier('e'),
            t.blockStatement([
              t.expressionStatement(
                t.callExpression(
                  t.memberExpression(
                    t.identifier('console'),
                    t.identifier('error')
                  ),
                  [t.identifier('e')]
                )
              ),
            ])
          )
        );
        
        body.body = [tryStatement];
      },
    },
  };
};
`;

// 埋点插件示例
const trackingPlugin = `
const t = require('@babel/types');

module.exports = function() {
  return {
    name: 'tracking-plugin',
    visitor: {
      // 给函数添加埋点
      'FunctionDeclaration|FunctionExpression|ArrowFunctionExpression'(path) {
        const functionName = path.node.id?.name || 'anonymous';
        
        // 创建埋点代码
        const trackingCall = t.expressionStatement(
          t.callExpression(
            t.identifier('track'),
            [
              t.stringLiteral('function_called'),
              t.objectExpression([
                t.objectProperty(
                  t.identifier('name'),
                  t.stringLiteral(functionName)
                ),
              ]),
            ]
          )
        );
        
        // 插入到函数体开头
        if (path.node.body.type === 'BlockStatement') {
          path.node.body.body.unshift(trackingCall);
        }
      },
    },
  };
};
`;

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. Polyfill 体积过大
 *    - 使用 useBuiltIns: 'usage'
 *    - 配置合理的 targets
 *
 * 2. 重复的 helper 代码
 *    - 使用 @babel/plugin-transform-runtime
 *
 * 3. 全局污染
 *    - 库开发时使用 transform-runtime
 *    - 业务代码可以全局 polyfill
 *
 * 4. 配置不生效
 *    - 检查 .babelrc 和 babel.config.js 的区别
 *    - babel.config.js 用于 monorepo
 *
 * 5. 某些语法没有转换
 *    - 检查 targets 配置
 *    - 可能需要额外的 plugin
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: Babel 是如何工作的？
 * A:
 *    1. Parse：源码 → AST（@babel/parser）
 *    2. Transform：AST 转换（@babel/traverse + plugins）
 *    3. Generate：AST → 代码（@babel/generator）
 *
 * Q2: 如何开发一个 Babel 插件？
 * A:
 *    1. 分析输入输出的 AST 结构（astexplorer.net）
 *    2. 编写 visitor 处理相应节点
 *    3. 使用 @babel/types 创建/修改节点
 *    4. 测试插件
 *
 * Q3: @babel/preset-env 和 @babel/plugin-transform-runtime 的区别？
 * A:
 *    preset-env：
 *    - 转换语法 + 按需引入 polyfill
 *    - 污染全局
 *    - 适合业务项目
 *
 *    transform-runtime：
 *    - 复用 helper + 沙箱化 polyfill
 *    - 不污染全局
 *    - 适合库开发
 *
 * Q4: AST 有哪些应用场景？
 * A:
 *    - 代码转换（Babel）
 *    - 代码压缩（Terser）
 *    - 代码检查（ESLint）
 *    - 代码格式化（Prettier）
 *    - 自动埋点
 *    - 国际化提取
 *    - 依赖分析
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：自动国际化提取
 *
 * 需求：自动提取代码中的中文，生成国际化配置
 *
 * 思路：
 * 1. 遍历 StringLiteral 节点
 * 2. 检测是否包含中文
 * 3. 生成 key，替换为 t(key)
 * 4. 输出 locale 文件
 */

const i18nPlugin = `
module.exports = function({ types: t }) {
  const zhTexts = {};
  let index = 0;
  
  return {
    visitor: {
      StringLiteral(path) {
        const value = path.node.value;
        // 检测中文
        if (/[\\u4e00-\\u9fa5]/.test(value)) {
          const key = 'text_' + index++;
          zhTexts[key] = value;
          
          // 替换为 t(key)
          path.replaceWith(
            t.callExpression(t.identifier('t'), [t.stringLiteral(key)])
          );
        }
      },
    },
    post() {
      // 输出 locale 文件
      console.log(JSON.stringify(zhTexts, null, 2));
    },
  };
};
`;

/**
 * 🏢 场景 2：按需加载组件库
 *
 * import { Button } from 'antd';
 * ↓
 * import Button from 'antd/es/button';
 * import 'antd/es/button/style';
 */

const importTransformPlugin = `
module.exports = function({ types: t }) {
  return {
    visitor: {
      ImportDeclaration(path) {
        const source = path.node.source.value;
        if (source !== 'antd') return;
        
        const specifiers = path.node.specifiers;
        if (!specifiers.length) return;
        
        const newImports = specifiers
          .filter(s => t.isImportSpecifier(s))
          .map(s => {
            const name = s.imported.name;
            const kebabName = name.replace(/([A-Z])/g, '-$1').toLowerCase().slice(1);
            
            return [
              // import Button from 'antd/es/button'
              t.importDeclaration(
                [t.importDefaultSpecifier(t.identifier(name))],
                t.stringLiteral(\`antd/es/\${kebabName}\`)
              ),
              // import 'antd/es/button/style'
              t.importDeclaration(
                [],
                t.stringLiteral(\`antd/es/\${kebabName}/style\`)
              ),
            ];
          })
          .flat();
        
        path.replaceWithMultiple(newImports);
      },
    },
  };
};
`;

export {
  babelConfigExample,
  runtimePluginExample,
  removeConsolePlugin,
  autoTryCatchPlugin,
  trackingPlugin,
  i18nPlugin,
  importTransformPlugin,
};

