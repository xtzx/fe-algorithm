/**
 * Vite 插件：环境变量替换
 *
 * 功能：
 * 1. 在代码中使用自定义格式的变量（如 __APP_VERSION__）
 * 2. 构建时自动替换为实际值
 * 3. 支持在 HTML 中替换
 *
 * 使用方法：
 *   import envReplacePlugin from './vite-plugins/env-replace-plugin';
 *
 *   export default defineConfig({
 *     plugins: [
 *       envReplacePlugin({
 *         replacements: {
 *           APP_VERSION: '1.0.0',
 *           BUILD_TIME: new Date().toISOString(),
 *           API_URL: 'https://api.example.com'
 *         }
 *       })
 *     ]
 *   });
 *
 * 代码中使用：
 *   console.log(__APP_VERSION__);  // → '1.0.0'
 *   console.log(__BUILD_TIME__);   // → '2024-01-01T00:00:00.000Z'
 */

import type { Plugin, ResolvedConfig } from 'vite';

// ==========================================
// 类型定义
// ==========================================

interface EnvReplaceOptions {
  /**
   * 要替换的变量映射
   * key: 变量名（不含前后缀）
   * value: 替换值
   */
  replacements?: Record<string, string | number | boolean>;

  /**
   * 变量前缀
   * @default '__'
   */
  prefix?: string;

  /**
   * 变量后缀
   * @default '__'
   */
  suffix?: string;

  /**
   * 是否在开发模式启用
   * @default true
   */
  enableInDev?: boolean;

  /**
   * 排除的文件模式
   * @default /node_modules/
   */
  exclude?: RegExp;
}

// ==========================================
// 插件实现
// ==========================================

export default function envReplacePlugin(options: EnvReplaceOptions = {}): Plugin {
  const {
    replacements = {},
    prefix = '__',
    suffix = '__',
    enableInDev = true,
    exclude = /node_modules/
  } = options;

  // 预处理替换规则
  const processedReplacements: Record<string, string> = {};

  for (const [key, value] of Object.entries(replacements)) {
    const pattern = `${prefix}${key}${suffix}`;
    // 值需要序列化为 JavaScript 表达式
    processedReplacements[pattern] = JSON.stringify(value);
  }

  // 存储配置
  let config: ResolvedConfig;
  let isDev: boolean;

  return {
    name: 'vite-plugin-env-replace',

    // ==========================================
    // 配置阶段
    // ==========================================

    /**
     * 修改 Vite 配置
     * 使用 Vite 的 define 功能进行替换（更高效）
     */
    config() {
      return {
        define: processedReplacements
      };
    },

    /**
     * 读取最终配置
     */
    configResolved(resolvedConfig) {
      config = resolvedConfig;
      isDev = config.command === 'serve';
    },

    // ==========================================
    // 代码转换（备用方案）
    // ==========================================

    /**
     * 手动转换代码
     * 这是 define 的备用方案，用于处理特殊情况
     */
    transform(code, id) {
      // 开发模式下且禁用了开发模式转换
      if (isDev && !enableInDev) {
        return;
      }

      // 排除 node_modules
      if (exclude && exclude.test(id)) {
        return;
      }

      // 检查是否需要替换
      let hasReplacement = false;
      for (const pattern of Object.keys(processedReplacements)) {
        if (code.includes(pattern)) {
          hasReplacement = true;
          break;
        }
      }

      if (!hasReplacement) {
        return;
      }

      // 执行替换
      let transformedCode = code;
      for (const [pattern, value] of Object.entries(processedReplacements)) {
        // 使用全局替换
        transformedCode = transformedCode.split(pattern).join(value);
      }

      return {
        code: transformedCode,
        // 不生成 sourcemap（简化处理）
        map: null
      };
    },

    // ==========================================
    // HTML 转换
    // ==========================================

    /**
     * 转换 index.html
     * 支持在 HTML 中使用变量
     */
    transformIndexHtml(html) {
      let transformedHtml = html;

      for (const [pattern, value] of Object.entries(processedReplacements)) {
        // HTML 中不需要 JSON 引号
        const rawValue = JSON.parse(value);
        transformedHtml = transformedHtml.split(pattern).join(String(rawValue));
      }

      return transformedHtml;
    }
  };
}


// ==========================================
// 高级用法：从环境变量自动读取
// ==========================================

/**
 * 自动从 process.env 读取带特定前缀的变量
 */
export function envAutoReplacePlugin(envPrefix: string = 'APP_'): Plugin {
  // 收集环境变量
  const replacements: Record<string, string> = {};

  for (const [key, value] of Object.entries(process.env)) {
    if (key.startsWith(envPrefix) && value !== undefined) {
      // 移除前缀，转换为 __KEY__ 格式
      const varName = key.substring(envPrefix.length);
      replacements[varName] = value;
    }
  }

  // 添加内置变量
  replacements['BUILD_TIME'] = new Date().toISOString();
  replacements['BUILD_MODE'] = process.env.NODE_ENV || 'development';

  return envReplacePlugin({ replacements });
}


// ==========================================
// 使用示例
// ==========================================

/*
// vite.config.ts
import { defineConfig } from 'vite';
import envReplacePlugin from './vite-plugins/env-replace-plugin';

export default defineConfig({
  plugins: [
    envReplacePlugin({
      replacements: {
        APP_NAME: 'My Awesome App',
        APP_VERSION: '1.0.0',
        BUILD_TIME: new Date().toISOString(),
        API_BASE_URL: process.env.API_URL || 'http://localhost:3000'
      }
    })
  ]
});

// 在代码中使用
// src/main.ts
console.log('App:', __APP_NAME__);
console.log('Version:', __APP_VERSION__);
console.log('Build Time:', __BUILD_TIME__);

fetch(`${__API_BASE_URL__}/api/users`);

// 在 HTML 中使用
// index.html
<title>__APP_NAME__</title>
<meta name="version" content="__APP_VERSION__">
*/

