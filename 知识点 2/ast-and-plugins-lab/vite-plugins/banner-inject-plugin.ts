/**
 * Vite 插件：Banner 注入
 *
 * 功能：
 * 1. 在构建产物的 JS/CSS 文件开头添加 banner 注释
 * 2. 可自定义 banner 内容
 * 3. 可配置包含/排除的文件模式
 *
 * 使用方法：
 *   import bannerInjectPlugin from './vite-plugins/banner-inject-plugin';
 *
 *   export default defineConfig({
 *     plugins: [
 *       bannerInjectPlugin({
 *         banner: '/** My App v1.0.0 */'
 *       })
 *     ]
 *   });
 */

import type { Plugin, OutputBundle, OutputChunk, OutputAsset } from 'vite';

// ==========================================
// 类型定义
// ==========================================

interface BannerInjectOptions {
  /**
   * Banner 内容
   * 可以是字符串，或返回字符串的函数
   */
  banner?: string | (() => string);

  /**
   * 包含的文件模式
   * @default /\.(js|css)$/
   */
  include?: RegExp;

  /**
   * 排除的文件模式
   * @default /node_modules/
   */
  exclude?: RegExp;
}

// ==========================================
// 默认 Banner 模板
// ==========================================

const defaultBanner = () => `/**
 * @license
 * Built with Vite
 * Build Time: ${new Date().toISOString()}
 */
`;

// ==========================================
// 插件实现
// ==========================================

export default function bannerInjectPlugin(options: BannerInjectOptions = {}): Plugin {
  // 合并配置
  const {
    banner = defaultBanner,
    include = /\.(js|css)$/,
    exclude = /node_modules/
  } = options;

  // 获取 banner 内容
  const getBanner = (): string => {
    if (typeof banner === 'function') {
      return banner();
    }
    return banner;
  };

  return {
    // 插件名称（必须）
    name: 'vite-plugin-banner-inject',

    // 只在构建模式下生效
    apply: 'build',

    // 使用 Rollup 的 generateBundle 钩子
    // 这个钩子在所有代码生成完成后、写入磁盘前调用
    generateBundle(_outputOptions, bundle: OutputBundle) {
      // 获取 banner 内容
      const bannerContent = getBanner();

      // 确保 banner 以换行结尾
      const normalizedBanner = bannerContent.endsWith('\n')
        ? bannerContent
        : bannerContent + '\n';

      // 遍历所有输出文件
      for (const [fileName, chunk] of Object.entries(bundle)) {
        // 检查是否匹配 include 模式
        if (!include.test(fileName)) {
          continue;
        }

        // 检查是否匹配 exclude 模式
        if (exclude && exclude.test(fileName)) {
          continue;
        }

        // 处理代码 chunk
        if (chunk.type === 'chunk') {
          const outputChunk = chunk as OutputChunk;
          outputChunk.code = normalizedBanner + outputChunk.code;
        }

        // 处理静态资源（如 CSS）
        if (chunk.type === 'asset') {
          const outputAsset = chunk as OutputAsset;
          const source = outputAsset.source;

          // 只处理文本内容
          if (typeof source === 'string') {
            outputAsset.source = normalizedBanner + source;
          }
        }
      }
    },

    // 可选：在开发模式下也添加 banner（调试用）
    // transform(code, id) {
    //   if (!include.test(id)) return;
    //   if (exclude && exclude.test(id)) return;
    //
    //   const bannerContent = getBanner();
    //   return {
    //     code: bannerContent + code,
    //     map: null
    //   };
    // }
  };
}


// ==========================================
// 高级用法示例
// ==========================================

/**
 * 带元数据的 Banner
 */
export function bannerWithMetadata(meta: {
  name: string;
  version: string;
  author?: string;
}): Plugin {
  return bannerInjectPlugin({
    banner: () => `/**
 * ${meta.name} v${meta.version}
 * ${meta.author ? `Author: ${meta.author}` : ''}
 * Build: ${new Date().toISOString()}
 *
 * This file is auto-generated. Do not edit.
 */
`
  });
}

/**
 * 从 package.json 读取信息的 Banner
 */
export function bannerFromPackage(): Plugin {
  let packageInfo: { name: string; version: string } | null = null;

  return {
    name: 'vite-plugin-banner-from-package',
    apply: 'build',

    // 在 configResolved 阶段读取 package.json
    async configResolved(config) {
      try {
        const fs = await import('fs');
        const path = await import('path');
        const packagePath = path.resolve(config.root, 'package.json');
        const packageContent = fs.readFileSync(packagePath, 'utf-8');
        packageInfo = JSON.parse(packageContent);
      } catch (e) {
        console.warn('Failed to read package.json');
      }
    },

    generateBundle(_options, bundle) {
      const banner = `/**
 * ${packageInfo?.name || 'Unknown'} v${packageInfo?.version || '0.0.0'}
 * Built at ${new Date().toISOString()}
 */
`;

      for (const chunk of Object.values(bundle)) {
        if (chunk.type === 'chunk' && chunk.fileName.endsWith('.js')) {
          chunk.code = banner + chunk.code;
        }
      }
    }
  };
}

