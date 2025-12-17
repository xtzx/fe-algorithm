/**
 * Rolldown 配置示例 (预览版)
 *
 * 注意: Rolldown 仍在开发中，此配置基于其设计目标和 Rollup 兼容性
 * API 可能会变化，请以官方文档为准
 *
 * Rolldown 的目标是与 Rollup 配置尽可能兼容
 */

import type { RollupOptions } from 'rollup';

// Rolldown 配置与 Rollup 基本兼容
const config: RollupOptions = {
  // ============================================
  // 入口配置
  // ============================================
  input: {
    // 多入口配置
    main: './src/index.ts',
    utils: './src/utils/index.ts',
  },

  // 或单入口
  // input: './src/index.ts',

  // ============================================
  // 输出配置
  // ============================================
  output: [
    // ES Module 输出
    {
      dir: 'dist/esm',
      format: 'es',
      entryFileNames: '[name].mjs',
      chunkFileNames: 'chunks/[name]-[hash].mjs',
      sourcemap: true,
      // 保留模块结构 (适合库)
      preserveModules: false,
    },
    // CommonJS 输出
    {
      dir: 'dist/cjs',
      format: 'cjs',
      entryFileNames: '[name].cjs',
      chunkFileNames: 'chunks/[name]-[hash].cjs',
      sourcemap: true,
      exports: 'named',
    },
  ],

  // ============================================
  // 外部依赖
  // ============================================
  // 不打包这些依赖，由使用者提供
  external: [
    // 精确匹配
    'react',
    'react-dom',

    // 正则匹配
    /^@babel\/.*/,

    // 函数判断
    (id) => id.includes('node_modules'),
  ],

  // ============================================
  // 插件配置
  // ============================================
  // Rolldown 目标是兼容 Rollup 插件
  plugins: [
    // ============================================
    // 常用插件示例 (Rollup 插件，Rolldown 应该兼容)
    // ============================================

    // 解析 Node.js 模块
    // nodeResolve({
    //   extensions: ['.ts', '.tsx', '.js', '.jsx'],
    //   browser: true,
    // }),

    // CommonJS 转 ESM
    // commonjs(),

    // TypeScript 编译 (使用 SWC 或 esbuild)
    // swc(),
    // esbuild({ target: 'es2020' }),

    // JSON 导入支持
    // json(),

    // 别名
    // alias({
    //   entries: {
    //     '@': './src',
    //   },
    // }),

    // 自定义插件示例
    {
      name: 'custom-plugin',

      // 构建开始
      buildStart() {
        console.log('Build started...');
      },

      // 解析模块 ID
      resolveId(source, importer) {
        // 返回 null 继续使用默认解析
        // 返回字符串表示解析结果
        if (source === 'virtual-module') {
          return '\0virtual-module';
        }
        return null;
      },

      // 加载模块内容
      load(id) {
        if (id === '\0virtual-module') {
          return 'export default "Hello from virtual module!"';
        }
        return null;
      },

      // 转换模块代码
      transform(code, id) {
        // 可以在这里做代码转换
        if (id.endsWith('.custom')) {
          return {
            code: `export default ${JSON.stringify(code)}`,
            map: null,
          };
        }
        return null;
      },

      // Chunk 生成后
      renderChunk(code, chunk) {
        // 可以在这里添加 banner 等
        const banner = `/* Built with Rolldown */\n`;
        return { code: banner + code, map: null };
      },

      // 构建结束
      buildEnd() {
        console.log('Build completed!');
      },
    },
  ],

  // ============================================
  // Tree Shaking 配置
  // ============================================
  treeshake: {
    // 模块副作用
    moduleSideEffects: 'no-external',

    // 属性读取被视为有副作用
    propertyReadSideEffects: false,

    // 未使用的导出
    // 'smallest' | 'recommended' | 'safest'
    preset: 'recommended',
  },

  // ============================================
  // 其他配置
  // ============================================

  // 监听模式配置
  watch: {
    include: 'src/**',
    exclude: 'node_modules/**',
    clearScreen: false,
  },

  // 警告处理
  onwarn(warning, warn) {
    // 忽略某些警告
    if (warning.code === 'CIRCULAR_DEPENDENCY') {
      return;
    }
    warn(warning);
  },
};

export default config;

/*
 * ============================================
 * Rolldown vs Rollup 差异说明
 * ============================================
 *
 * 兼容的部分:
 * ✅ input/output 配置
 * ✅ external 配置
 * ✅ 标准插件钩子 (resolveId, load, transform, renderChunk 等)
 * ✅ treeshake 配置
 * ✅ 大部分 Rollup 插件
 *
 * 可能不兼容的部分:
 * ⚠️ 某些 Rollup 内部 API
 * ⚠️ this.getModuleInfo 等方法的细节
 * ⚠️ AST 操作 (Rolldown 使用不同的 AST)
 *
 * Rolldown 独有的优势:
 * 🚀 Rust 实现，多线程并行
 * 🚀 更快的解析和打包速度
 * 🚀 与 Vite 深度集成
 */

/*
 * ============================================
 * 使用方式 (预期)
 * ============================================
 *
 * # 安装 (待 Rolldown 发布)
 * npm install -D rolldown
 *
 * # 构建
 * npx rolldown -c rolldown.config.ts
 *
 * # 监听模式
 * npx rolldown -c rolldown.config.ts --watch
 */

