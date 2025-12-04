/**
 * ============================================================
 * 📚 构建工具原理（Webpack / Vite）
 * ============================================================
 *
 * 面试考察重点：
 * 1. Webpack 的核心概念和工作流程
 * 2. Vite 的优势和原理
 * 3. 常见优化手段
 * 4. Loader 和 Plugin 的区别
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 为什么需要构建工具？
 *
 * 1. 模块化支持：ESM、CommonJS、AMD
 * 2. 代码转换：TS → JS、Sass → CSS
 * 3. 性能优化：压缩、Tree Shaking、代码分割
 * 4. 开发体验：热更新、Source Map
 * 5. 兼容性：Polyfill、PostCSS
 */

// ============================================================
// 2. Webpack 核心概念
// ============================================================

/**
 * 📊 Webpack 五大核心概念
 *
 * 1. Entry：入口文件
 * 2. Output：输出配置
 * 3. Loader：文件转换器
 * 4. Plugin：扩展功能
 * 5. Mode：模式（development/production）
 */

const webpackConfigExample = `
// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  // 入口
  entry: {
    main: './src/index.js',
    admin: './src/admin.js',
  },
  
  // 输出
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].[contenthash].js',
    clean: true, // 清理旧文件
  },
  
  // 模式
  mode: 'production',
  
  // Loader
  module: {
    rules: [
      {
        test: /\\.js$/,
        exclude: /node_modules/,
        use: 'babel-loader',
      },
      {
        test: /\\.css$/,
        use: [MiniCssExtractPlugin.loader, 'css-loader', 'postcss-loader'],
      },
      {
        test: /\\.(png|jpg|gif)$/,
        type: 'asset',
        parser: {
          dataUrlCondition: {
            maxSize: 8 * 1024, // 8KB 以下转 base64
          },
        },
      },
    ],
  },
  
  // Plugin
  plugins: [
    new HtmlWebpackPlugin({
      template: './public/index.html',
    }),
    new MiniCssExtractPlugin({
      filename: '[name].[contenthash].css',
    }),
  ],
  
  // 优化
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendors: {
          test: /[\\\\/]node_modules[\\\\/]/,
          name: 'vendors',
          priority: 10,
        },
      },
    },
  },
};
`;

/**
 * 📊 Webpack 构建流程
 *
 * 1. 初始化：读取配置，创建 Compiler
 * 2. 编译：从 Entry 开始，递归分析依赖
 * 3. 构建模块：调用 Loader 转换文件
 * 4. 生成 Chunk：根据依赖关系组合模块
 * 5. 输出：生成最终文件
 *
 * ┌──────────────────────────────────────────────────────────────┐
 * │                                                              │
 * │  Entry ──► Loader ──► Module ──► Chunk ──► Bundle           │
 * │    │         │          │          │          │              │
 * │  入口      转换        模块      代码块      输出             │
 * │                         │                                    │
 * │                    Plugin (各阶段钩子)                       │
 * │                                                              │
 * └──────────────────────────────────────────────────────────────┘
 */

/**
 * 📊 Loader vs Plugin
 *
 * Loader：
 * - 文件转换器
 * - 作用于单个文件
 * - 链式调用，从右到左
 * - 例：babel-loader、css-loader
 *
 * Plugin：
 * - 扩展 Webpack 功能
 * - 作用于整个构建流程
 * - 基于 Tapable 事件系统
 * - 例：HtmlWebpackPlugin、MiniCssExtractPlugin
 *
 * ⚠️ 易错点：
 * - Loader 顺序是从右到左执行
 * - css-loader 解析 CSS，style-loader 注入 DOM
 */

// 简单的 Loader 实现
const simpleLoader = `
// my-loader.js
module.exports = function(source) {
  // source 是文件内容
  // 返回处理后的内容
  return source.replace(/console\\.log\\(.*?\\);?/g, '');
};

// 异步 Loader
module.exports = function(source) {
  const callback = this.async();
  
  someAsyncOperation(source, (err, result) => {
    if (err) return callback(err);
    callback(null, result);
  });
};
`;

// 简单的 Plugin 实现
const simplePlugin = `
// my-plugin.js
class MyPlugin {
  apply(compiler) {
    // 注册钩子
    compiler.hooks.emit.tapAsync('MyPlugin', (compilation, callback) => {
      // compilation 包含所有编译信息
      const assets = compilation.assets;
      
      // 添加一个文件
      assets['filelist.txt'] = {
        source: () => Object.keys(assets).join('\\n'),
        size: () => Object.keys(assets).join('\\n').length,
      };
      
      callback();
    });
  }
}

module.exports = MyPlugin;
`;

// ============================================================
// 3. Vite 原理
// ============================================================

/**
 * 📊 Vite 的优势
 *
 * 1. 极速冷启动：不需要打包，直接启动
 * 2. 即时热更新：基于 ESM，只更新修改的模块
 * 3. 按需编译：请求时才编译
 *
 * 📊 Vite vs Webpack 开发模式对比
 *
 * Webpack：
 * ┌─────────────────────────────────────────────────────────────┐
 * │  Entry ──► 分析依赖 ──► 打包所有模块 ──► Bundle ──► 启动    │
 * │                        （耗时！）                           │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Vite：
 * ┌─────────────────────────────────────────────────────────────┐
 * │  启动服务器 ──► 浏览器请求 ──► 按需编译 ──► 返回              │
 * │  （极快！）                                                  │
 * └─────────────────────────────────────────────────────────────┘
 */

/**
 * 📊 Vite 工作原理
 *
 * 开发模式：
 * 1. 启动 Koa 服务器
 * 2. 浏览器请求模块时实时编译
 * 3. 利用浏览器原生 ESM
 * 4. 依赖预构建（esbuild）
 *
 * 生产模式：
 * 1. 使用 Rollup 打包
 * 2. 代码分割、压缩等优化
 */

/**
 * 📊 依赖预构建
 *
 * 为什么需要预构建？
 * 1. 将 CommonJS/UMD 转换为 ESM
 * 2. 合并小模块，减少请求数
 *
 * 例：lodash-es 有 600+ 模块，预构建后只有 1 个
 *
 * 存储位置：node_modules/.vite
 */

const viteConfigExample = `
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  
  // 依赖预构建
  optimizeDeps: {
    include: ['lodash-es'], // 强制预构建
    exclude: ['some-package'], // 排除预构建
  },
  
  // 开发服务器
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  
  // 构建配置
  build: {
    target: 'es2015',
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
});
`;

// ============================================================
// 4. 构建优化
// ============================================================

/**
 * 📊 常见优化手段
 *
 * 1. 代码分割（Code Splitting）
 *    - 入口分割
 *    - 动态导入
 *    - 公共代码提取
 *
 * 2. Tree Shaking
 *    - 移除未使用代码
 *    - 需要 ESM
 *    - sideEffects 配置
 *
 * 3. 缓存
 *    - 持久化缓存（Webpack 5）
 *    - contenthash 文件名
 *
 * 4. 并行处理
 *    - thread-loader
 *    - parallel-webpack
 *
 * 5. 减少搜索范围
 *    - resolve.alias
 *    - resolve.extensions
 *    - exclude/include
 */

const optimizationConfig = `
// webpack.config.js 优化配置
module.exports = {
  // 1. 持久化缓存（Webpack 5）
  cache: {
    type: 'filesystem',
  },
  
  // 2. 代码分割
  optimization: {
    splitChunks: {
      chunks: 'all',
      minSize: 20000,
      cacheGroups: {
        vendors: {
          test: /[\\\\/]node_modules[\\\\/]/,
          name: 'vendors',
          priority: 10,
        },
        react: {
          test: /[\\\\/]node_modules[\\\\/](react|react-dom)[\\\\/]/,
          name: 'react',
          priority: 20,
        },
        common: {
          minChunks: 2,
          name: 'common',
          priority: 5,
        },
      },
    },
  },
  
  // 3. 减少搜索范围
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
    extensions: ['.js', '.jsx', '.ts', '.tsx'],
    modules: [path.resolve(__dirname, 'node_modules')],
  },
  
  // 4. 并行处理
  module: {
    rules: [
      {
        test: /\\.js$/,
        use: [
          'thread-loader',
          'babel-loader',
        ],
      },
    ],
  },
};
`;

/**
 * 📊 Tree Shaking 原理
 *
 * 基于 ESM 的静态分析：
 * - ESM 的 import/export 是静态的
 * - 编译时就能确定哪些代码被使用
 * - 未使用的代码在生产构建时移除
 *
 * 前提条件：
 * - 使用 ESM（不能是 CommonJS）
 * - 使用 production 模式
 * - package.json 配置 sideEffects
 */

const treeshakingExample = `
// package.json
{
  "sideEffects": false  // 所有模块都是纯的
}

// 或者指定有副作用的文件
{
  "sideEffects": [
    "*.css",
    "*.scss",
    "./src/polyfill.js"
  ]
}

// ⚠️ 常见问题：第三方库不支持 Tree Shaking
// 解决：使用支持 ESM 的版本，如 lodash-es
import { debounce } from 'lodash-es'; // ✅ 可以 Tree Shaking
import _ from 'lodash'; // ❌ 无法 Tree Shaking
`;

// ============================================================
// 5. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. Loader 顺序错误
 *    - 从右到左执行
 *    - ['style-loader', 'css-loader'] 先 css-loader
 *
 * 2. contenthash vs chunkhash
 *    - contenthash：根据内容生成
 *    - chunkhash：根据 chunk 生成
 *    - 推荐：JS 用 contenthash，CSS 用 contenthash
 *
 * 3. Tree Shaking 失效
 *    - 使用了 CommonJS
 *    - 没有配置 sideEffects
 *    - 代码有副作用
 *
 * 4. 开发/生产配置混淆
 *    - 开发：source-map, HMR
 *    - 生产：压缩, 代码分割
 *
 * 5. 循环依赖
 *    - 导致运行时错误
 *    - 使用工具检测：circular-dependency-plugin
 */

// ============================================================
// 6. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: Webpack 的 HMR 原理？
 * A:
 *    1. 文件变化，Webpack 重新编译
 *    2. 生成新的 hash 和更新 manifest
 *    3. 通过 WebSocket 通知浏览器
 *    4. 浏览器请求更新的模块
 *    5. 执行 module.hot.accept 回调
 *
 * Q2: Vite 为什么比 Webpack 快？
 * A:
 *    开发模式：
 *    - 不需要打包，利用浏览器 ESM
 *    - 按需编译
 *    - 依赖预构建用 esbuild（Go 语言，快 10-100 倍）
 *
 *    生产模式：
 *    - 使用 Rollup，速度相近
 *
 * Q3: 如何分析打包体积？
 * A:
 *    - webpack-bundle-analyzer：可视化分析
 *    - source-map-explorer：分析 source map
 *    - 关注：大文件、重复依赖、未使用代码
 *
 * Q4: 如何实现按需加载？
 * A:
 *    - 动态 import()：import('./module').then()
 *    - React.lazy：懒加载组件
 *    - 路由懒加载
 *    - webpack magic comments
 */

// ============================================================
// 7. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：首次构建慢
 *
 * 分析：
 * - 项目大，模块多
 * - 没有利用缓存
 *
 * 解决：
 * 1. 开启持久化缓存（cache: { type: 'filesystem' }）
 * 2. 使用 thread-loader 并行编译
 * 3. 缩小构建范围（include/exclude）
 * 4. 开发环境不压缩
 */

/**
 * 🏢 场景 2：打包体积大
 *
 * 分析：
 * - webpack-bundle-analyzer 分析
 * - 找出大文件和重复依赖
 *
 * 解决：
 * 1. Tree Shaking + sideEffects
 * 2. 代码分割
 * 3. 按需引入（lodash → lodash-es）
 * 4. 外部化大依赖（externals）
 * 5. 压缩（terser、gzip）
 */

/**
 * 🏢 场景 3：HMR 不生效
 *
 * 可能原因：
 * - 没有 module.hot.accept
 * - 组件没有默认导出
 * - 配置问题
 *
 * 解决：
 * - React：使用 @pmmmwh/react-refresh-webpack-plugin
 * - Vue：vue-loader 自带支持
 */

export {
  webpackConfigExample,
  viteConfigExample,
  simpleLoader,
  simplePlugin,
  optimizationConfig,
  treeshakingExample,
};

