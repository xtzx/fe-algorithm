/**
 * ============================================================
 * 📚 CSS 工程化
 * ============================================================
 *
 * 面试考察重点：
 * 1. CSS 模块化方案对比
 * 2. CSS 预处理器
 * 3. PostCSS 与构建工具
 * 4. 设计系统与 Design Token
 * 5. 主题切换方案
 */

// ============================================================
// 1. CSS 模块化方案对比
// ============================================================

/**
 * 📊 主流 CSS 方案对比
 *
 * ┌─────────────────┬─────────────────────────────────────────────────────────┐
 * │ 方案             │ 特点                                                    │
 * ├─────────────────┼─────────────────────────────────────────────────────────┤
 * │ 普通 CSS         │ 全局作用域，容易冲突                                     │
 * │ BEM 命名规范     │ 约定式隔离，需要团队遵守                                 │
 * │ CSS Modules     │ 编译时生成唯一类名，零运行时                             │
 * │ CSS-in-JS       │ 运行时生成，可使用 JS 逻辑，有运行时开销                  │
 * │ 原子化 CSS       │ 单一职责类名，复用性强，学习成本高                        │
 * └─────────────────┴─────────────────────────────────────────────────────────┘
 */

/**
 * 📊 方案 1：BEM 命名规范
 *
 * Block（块）+ Element（元素）+ Modifier（修饰符）
 * 格式：block__element--modifier
 */
const bemExample = `
  /* Block: 独立组件 */
  .card {}
  
  /* Element: 组件的一部分 */
  .card__header {}
  .card__body {}
  .card__footer {}
  
  /* Modifier: 状态或变体 */
  .card--primary {}
  .card--disabled {}
  .card__header--large {}
`;

/**
 * 📊 方案 2：CSS Modules
 *
 * 编译时为每个类名生成唯一的标识符
 */
const cssModulesExample = `
  /* styles.module.css */
  .title {
    font-size: 24px;
  }
  .button {
    padding: 10px 20px;
  }
  
  /* 编译后 */
  .styles_title_x7s2 {
    font-size: 24px;
  }
  
  /* React 中使用 */
  // import styles from './styles.module.css';
  // <h1 className={styles.title}>Hello</h1>
  
  /* 组合样式 */
  .title {
    composes: base from './base.module.css';
    font-size: 24px;
  }
`;

/**
 * 📊 方案 3：CSS-in-JS
 *
 * 在 JavaScript 中编写 CSS
 * 代表：styled-components、Emotion、Stitches
 */
const cssInJsExample = `
  // styled-components
  const Button = styled.button\`
    padding: 10px 20px;
    background: \${props => props.primary ? 'blue' : 'gray'};
    color: white;
    
    &:hover {
      opacity: 0.8;
    }
    
    \${props => props.large && css\`
      padding: 15px 30px;
      font-size: 18px;
    \`}
  \`;
  
  // 使用
  // <Button primary large>Click</Button>
  
  // Emotion
  const buttonStyle = css\`
    padding: 10px 20px;
    background: blue;
  \`;
  // <button css={buttonStyle}>Click</button>
`;

/**
 * 📊 方案 4：原子化 CSS（Atomic CSS / Utility-First）
 *
 * 代表：Tailwind CSS、UnoCSS、Windi CSS
 */
const atomicCssExample = `
  <!-- Tailwind CSS -->
  <div class="flex items-center justify-between p-4 bg-white rounded-lg shadow-md">
    <h1 class="text-xl font-bold text-gray-900">Title</h1>
    <button class="px-4 py-2 text-white bg-blue-500 rounded hover:bg-blue-600">
      Click
    </button>
  </div>
  
  <!-- 优点 -->
  - 几乎零 CSS 体积增长（高度复用）
  - 无需命名
  - 样式与标记紧密关联
  
  <!-- 缺点 -->
  - HTML 较臃肿
  - 学习成本
  - 某些复杂样式难以实现
`;

/**
 * 📊 方案选型建议
 *
 * 小型项目：普通 CSS + BEM / CSS Modules
 * 中型项目：CSS Modules / Tailwind CSS
 * 大型项目：CSS-in-JS（需要动态样式）/ 原子化 CSS（追求性能）
 * 组件库：CSS-in-JS（styled-components）/ CSS Modules
 *
 * 💡 追问：CSS-in-JS 的优缺点？
 *
 * 优点：
 * - 样式与组件共存，便于维护
 * - 自动处理作用域
 * - 可使用 JS 逻辑（变量、条件、循环）
 * - 支持主题切换
 *
 * 缺点：
 * - 运行时开销
 * - 包体积增加
 * - SSR 需要额外配置
 * - 调试相对困难
 */

// ============================================================
// 2. CSS 预处理器
// ============================================================

/**
 * 📊 主流预处理器
 *
 * Sass/SCSS：最流行，功能强大
 * Less：语法接近 CSS，学习成本低
 * Stylus：语法灵活，Python 风格
 */

const scssFeatures = `
  // 1. 变量
  $primary-color: #1890ff;
  $spacing: 8px;
  
  .button {
    background: $primary-color;
    padding: $spacing * 2;
  }
  
  // 2. 嵌套
  .nav {
    ul {
      margin: 0;
      li {
        display: inline-block;
        a {
          color: $primary-color;
          &:hover {
            text-decoration: underline;
          }
        }
      }
    }
  }
  
  // 3. Mixin（混入）
  @mixin flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
  }
  
  @mixin ellipsis($lines: 1) {
    @if $lines == 1 {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    } @else {
      display: -webkit-box;
      -webkit-line-clamp: $lines;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
  }
  
  .card {
    @include flex-center;
    .title {
      @include ellipsis(2);
    }
  }
  
  // 4. 继承
  %button-base {
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
  }
  
  .btn-primary {
    @extend %button-base;
    background: $primary-color;
  }
  
  // 5. 函数
  @function rem($px) {
    @return $px / 16 * 1rem;
  }
  
  .text {
    font-size: rem(24);  // 1.5rem
  }
  
  // 6. 循环
  @for $i from 1 through 5 {
    .mt-#{$i} {
      margin-top: $i * 8px;
    }
  }
  
  // 7. 条件
  @mixin theme($mode) {
    @if $mode == dark {
      background: #1a1a1a;
      color: #fff;
    } @else {
      background: #fff;
      color: #333;
    }
  }
`;

// ============================================================
// 3. PostCSS
// ============================================================

/**
 * 📖 什么是 PostCSS？
 *
 * PostCSS 是一个 CSS 处理工具，通过插件系统转换 CSS。
 * 它本身只是一个平台，功能由插件提供。
 *
 * 📊 常用插件
 *
 * autoprefixer - 自动添加浏览器前缀
 * postcss-preset-env - 使用未来 CSS 语法
 * postcss-pxtorem - px 转 rem
 * postcss-px-to-viewport - px 转 vw
 * cssnano - CSS 压缩
 * postcss-import - @import 内联
 * postcss-nested - 支持嵌套语法
 */

const postcssConfig = `
  // postcss.config.js
  module.exports = {
    plugins: [
      require('postcss-import'),
      require('postcss-nested'),
      require('autoprefixer'),
      require('postcss-preset-env')({
        stage: 3,
        features: {
          'nesting-rules': true,
          'custom-media-queries': true,
        }
      }),
      require('cssnano')({
        preset: 'default',
      }),
    ]
  };
  
  // 移动端适配
  // postcss.config.js
  module.exports = {
    plugins: {
      'postcss-px-to-viewport': {
        viewportWidth: 750,
        unitPrecision: 5,
        viewportUnit: 'vw',
        selectorBlackList: [],
        minPixelValue: 1,
        mediaQuery: false,
      }
    }
  };
`;

// ============================================================
// 4. 设计系统与 Design Token
// ============================================================

/**
 * 📖 什么是 Design Token？
 *
 * Design Token 是设计系统的原子单位，
 * 用于存储设计决策（颜色、间距、字体等）。
 *
 * 📊 Token 层级
 *
 * 1. 全局 Token（Global）：原始值
 *    --color-blue-500: #1890ff
 *
 * 2. 语义 Token（Semantic）：用途相关
 *    --color-primary: var(--color-blue-500)
 *
 * 3. 组件 Token（Component）：组件特定
 *    --button-bg-color: var(--color-primary)
 */

const designTokenExample = `
  /* 1. 全局 Token */
  :root {
    /* 颜色 */
    --color-gray-50: #fafafa;
    --color-gray-100: #f5f5f5;
    --color-gray-200: #e5e5e5;
    --color-gray-900: #171717;
    
    --color-blue-500: #1890ff;
    --color-blue-600: #096dd9;
    
    --color-red-500: #ff4d4f;
    --color-green-500: #52c41a;
    
    /* 间距 */
    --spacing-1: 4px;
    --spacing-2: 8px;
    --spacing-3: 12px;
    --spacing-4: 16px;
    --spacing-6: 24px;
    --spacing-8: 32px;
    
    /* 字体 */
    --font-size-xs: 12px;
    --font-size-sm: 14px;
    --font-size-base: 16px;
    --font-size-lg: 18px;
    --font-size-xl: 20px;
    --font-size-2xl: 24px;
    
    --font-weight-normal: 400;
    --font-weight-medium: 500;
    --font-weight-bold: 700;
    
    /* 圆角 */
    --radius-sm: 2px;
    --radius-base: 4px;
    --radius-lg: 8px;
    --radius-full: 9999px;
    
    /* 阴影 */
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
    --shadow-base: 0 4px 6px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
    
    /* 过渡 */
    --transition-fast: 150ms ease;
    --transition-base: 300ms ease;
  }
  
  /* 2. 语义 Token */
  :root {
    /* 颜色语义 */
    --color-primary: var(--color-blue-500);
    --color-primary-hover: var(--color-blue-600);
    --color-success: var(--color-green-500);
    --color-error: var(--color-red-500);
    
    --color-text-primary: var(--color-gray-900);
    --color-text-secondary: var(--color-gray-600);
    --color-text-disabled: var(--color-gray-400);
    
    --color-bg-page: var(--color-gray-50);
    --color-bg-container: #fff;
    --color-bg-elevated: #fff;
    
    --color-border: var(--color-gray-200);
  }
  
  /* 3. 组件 Token */
  :root {
    /* Button */
    --button-height-sm: 24px;
    --button-height-base: 32px;
    --button-height-lg: 40px;
    --button-padding-x: var(--spacing-4);
    --button-border-radius: var(--radius-base);
    
    /* Input */
    --input-height: 32px;
    --input-border-color: var(--color-border);
    --input-focus-border-color: var(--color-primary);
    
    /* Card */
    --card-padding: var(--spacing-4);
    --card-border-radius: var(--radius-lg);
    --card-shadow: var(--shadow-base);
  }
`;

// ============================================================
// 5. 主题切换
// ============================================================

/**
 * 📊 主题切换方案
 */

// 方案 1：CSS 变量 + 类名切换（推荐）
const themeWithCSSVariables = `
  /* 定义主题变量 */
  :root {
    /* 亮色主题（默认） */
    --color-bg: #ffffff;
    --color-text: #333333;
    --color-primary: #1890ff;
  }
  
  /* 暗色主题 */
  [data-theme="dark"] {
    --color-bg: #1a1a1a;
    --color-text: #ffffff;
    --color-primary: #177ddc;
  }
  
  /* 使用变量 */
  body {
    background-color: var(--color-bg);
    color: var(--color-text);
  }
  
  /* JavaScript 切换 */
  // document.documentElement.setAttribute('data-theme', 'dark');
`;

// 方案 2：prefers-color-scheme（跟随系统）
const themeWithMediaQuery = `
  :root {
    --color-bg: #ffffff;
    --color-text: #333333;
  }
  
  @media (prefers-color-scheme: dark) {
    :root {
      --color-bg: #1a1a1a;
      --color-text: #ffffff;
    }
  }
  
  /* JavaScript 检测 */
  // const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  /* 监听变化 */
  // window.matchMedia('(prefers-color-scheme: dark)')
  //   .addEventListener('change', e => {
  //     console.log(e.matches ? 'dark' : 'light');
  //   });
`;

// 方案 3：CSS-in-JS 主题
const themeWithCSSinJS = `
  // styled-components
  const lightTheme = {
    colors: {
      bg: '#ffffff',
      text: '#333333',
      primary: '#1890ff',
    }
  };
  
  const darkTheme = {
    colors: {
      bg: '#1a1a1a',
      text: '#ffffff',
      primary: '#177ddc',
    }
  };
  
  // 使用 ThemeProvider
  // <ThemeProvider theme={isDark ? darkTheme : lightTheme}>
  //   <App />
  // </ThemeProvider>
  
  // 组件中使用
  // const Button = styled.button\`
  //   background: \${props => props.theme.colors.primary};
  // \`;
`;

// ============================================================
// 6. 高频面试题
// ============================================================

/**
 * 题目 1：CSS 样式隔离方案有哪些？优缺点是什么？
 *
 * 1. BEM 命名规范
 *    优点：无构建依赖，团队协作清晰
 *    缺点：需要人工遵守，类名较长
 *
 * 2. CSS Modules
 *    优点：自动生成唯一类名，零运行时
 *    缺点：需要构建工具支持，动态样式较麻烦
 *
 * 3. CSS-in-JS
 *    优点：与组件共存，支持动态样式
 *    缺点：运行时开销，包体积增加
 *
 * 4. Shadow DOM
 *    优点：浏览器原生隔离
 *    缺点：样式难以穿透，兼容性
 */

/**
 * 题目 2：如何实现换肤功能？
 *
 * 推荐：CSS 变量 + 类名切换
 *
 * 1. 定义主题变量
 * 2. 使用 data-theme 属性切换
 * 3. 可选：支持 prefers-color-scheme 跟随系统
 * 4. 本地存储用户偏好
 *
 * 注意事项：
 * - 图片/图标也需要适配
 * - 考虑过渡动画
 * - 考虑首屏闪烁问题（FOUC）
 */

/**
 * 题目 3：CSS 变量和 SCSS 变量的区别？
 *
 * SCSS 变量：
 * - 编译时确定，生成静态 CSS
 * - 不能在运行时修改
 * - 有作用域
 *
 * CSS 变量：
 * - 运行时生效
 * - 可以通过 JS 动态修改
 * - 可以继承和覆盖
 * - 可以响应媒体查询
 *
 * 建议：构建时用 SCSS，运行时动态用 CSS 变量
 */

export {
  bemExample,
  cssModulesExample,
  cssInJsExample,
  atomicCssExample,
  scssFeatures,
  postcssConfig,
  designTokenExample,
  themeWithCSSVariables,
  themeWithMediaQuery,
  themeWithCSSinJS,
};

