/**
 * Next.js + Turbopack 配置示例
 *
 * 启动方式: next dev --turbo
 *
 * 注意: Turbopack 配置是 Next.js 配置的一部分
 */

import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // ============================================
  // Turbopack 相关配置 (experimental)
  // ============================================
  experimental: {
    // Turbopack 专用配置
    turbo: {
      // ============================================
      // Loader 配置
      // ============================================
      // 类似 Webpack 的 loader 概念
      // 目前支持有限，主要用于特殊文件类型
      loaders: {
        // SVG 文件使用 @svgr/webpack 处理
        '.svg': ['@svgr/webpack'],

        // GraphQL 文件处理
        '.graphql': ['graphql-tag/loader'],

        // Markdown 文件处理
        '.md': ['raw-loader'],
      },

      // ============================================
      // 解析别名
      // ============================================
      // 类似 Webpack 的 resolve.alias
      resolveAlias: {
        // 路径别名
        '@': './src',
        '@components': './src/components',
        '@utils': './src/utils',
        '@hooks': './src/hooks',

        // 模块别名 (用于替换某些包)
        // 例如: 用轻量级实现替换重量级库
        // 'lodash': 'lodash-es',
      },

      // ============================================
      // 解析扩展名
      // ============================================
      // 文件扩展名解析顺序
      resolveExtensions: [
        '.tsx',
        '.ts',
        '.jsx',
        '.js',
        '.mjs',
        '.json',
      ],
    },

    // ============================================
    // 其他实验性功能 (可能与 Turbopack 相关)
    // ============================================

    // Server Actions (已稳定)
    // serverActions: true,

    // PPR (Partial Prerendering)
    // ppr: true,
  },

  // ============================================
  // 编译器配置 (SWC)
  // ============================================
  // 这部分配置影响 SWC 编译器行为
  compiler: {
    // 生产环境移除 console
    removeConsole: process.env.NODE_ENV === 'production',

    // 移除 React 属性 (如 data-testid)
    reactRemoveProperties: process.env.NODE_ENV === 'production',

    // Emotion CSS-in-JS 支持
    emotion: true,

    // Styled Components 支持
    styledComponents: true,
  },

  // ============================================
  // 其他 Next.js 配置
  // ============================================

  // 严格模式
  reactStrictMode: true,

  // 图片优化
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.example.com',
      },
    ],
  },

  // 环境变量
  env: {
    CUSTOM_VAR: process.env.CUSTOM_VAR,
  },

  // 重定向
  async redirects() {
    return [
      {
        source: '/old-page',
        destination: '/new-page',
        permanent: true,
      },
    ];
  },

  // Headers
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
        ],
      },
    ];
  },
};

export default nextConfig;

/*
 * ============================================
 * 使用说明
 * ============================================
 *
 * 1. 开发模式启用 Turbopack:
 *    npx next dev --turbo
 *
 *    或在 package.json 中:
 *    {
 *      "scripts": {
 *        "dev": "next dev --turbo"
 *      }
 *    }
 *
 * 2. 当前限制 (截至 2024):
 *    - 生产构建仍在 Beta
 *    - Webpack 插件不兼容
 *    - 部分 loader 需要适配
 *
 * 3. 性能对比:
 *    - 冷启动: 比 Webpack 快 ~10x
 *    - HMR: 几乎即时 (~10ms)
 */

