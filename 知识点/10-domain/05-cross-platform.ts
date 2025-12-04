/**
 * ============================================================
 * 📚 跨端开发
 * ============================================================
 *
 * 面试考察重点：
 * 1. 跨端方案对比
 * 2. React Native 原理
 * 3. Flutter 原理
 * 4. 小程序原理
 */

// ============================================================
// 1. 跨端方案对比
// ============================================================

/**
 * 📊 跨端方案分类
 *
 * 1. Web 容器（Hybrid）
 *    - Cordova、Ionic
 *    - WebView 渲染
 *    - 性能差，但开发简单
 *
 * 2. 原生渲染
 *    - React Native、Weex
 *    - JS 逻辑 + 原生渲染
 *    - 性能中等
 *
 * 3. 自绘引擎
 *    - Flutter
 *    - Skia 引擎自绘
 *    - 性能好，一致性高
 *
 * 4. 小程序
 *    - 微信/支付宝/抖音小程序
 *    - 双线程架构
 *    - 平台受限
 *
 * 📊 方案对比
 *
 * ┌─────────────────┬──────────────┬──────────────┬──────────────┐
 * │ 维度            │ React Native │ Flutter      │ 小程序        │
 * ├─────────────────┼──────────────┼──────────────┼──────────────┤
 * │ 语言            │ JavaScript   │ Dart         │ JS + WXML    │
 * │ 渲染            │ 原生组件      │ 自绘(Skia)   │ WebView      │
 * │ 性能            │ 中           │ 高           │ 中低         │
 * │ 开发效率        │ 高           │ 中           │ 高           │
 * │ 热更新          │ 支持         │ 有限制       │ 支持         │
 * │ 生态            │ 丰富         │ 增长中       │ 平台内       │
 * └─────────────────┴──────────────┴──────────────┴──────────────┘
 */

// ============================================================
// 2. React Native 原理
// ============================================================

/**
 * 📊 RN 架构
 *
 * 旧架构（Bridge）：
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                     React Native                                │
 * │                                                                 │
 * │  ┌─────────────────┐                    ┌─────────────────┐    │
 * │  │   JS Thread     │                    │  Native Thread  │    │
 * │  │                 │                    │                 │    │
 * │  │  React 组件     │    Bridge          │  原生 View      │    │
 * │  │  业务逻辑       │ ◄──(JSON)────────► │  原生 Module    │    │
 * │  │                 │    异步            │                 │    │
 * │  └─────────────────┘                    └─────────────────┘    │
 * │                                                                 │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * 新架构（Fabric + TurboModule）：
 *
 * - JSI：直接调用 C++，无需序列化
 * - Fabric：新的渲染系统
 * - TurboModule：按需加载原生模块
 * - Codegen：类型安全的代码生成
 */

// RN 组件示例
const rnComponentExample = `
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

function Button({ title, onPress }) {
  return (
    <TouchableOpacity onPress={onPress} style={styles.button}>
      <Text style={styles.text}>{title}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  text: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default Button;
`;

// 原生模块桥接
const nativeModuleExample = `
// iOS - MyNativeModule.m
#import <React/RCTBridgeModule.h>

@interface MyNativeModule : NSObject <RCTBridgeModule>
@end

@implementation MyNativeModule

RCT_EXPORT_MODULE();

RCT_EXPORT_METHOD(showToast:(NSString *)message) {
  dispatch_async(dispatch_get_main_queue(), ^{
    // 显示 Toast
  });
}

@end

// JavaScript 调用
import { NativeModules } from 'react-native';
const { MyNativeModule } = NativeModules;

MyNativeModule.showToast('Hello from RN!');
`;

// ============================================================
// 3. Flutter 原理
// ============================================================

/**
 * 📊 Flutter 架构
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        Flutter                                  │
 * │                                                                 │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │                   Framework (Dart)                       │   │
 * │  │  Material / Cupertino                                    │   │
 * │  │  Widgets                                                 │   │
 * │  │  Rendering                                               │   │
 * │  │  Animation / Painting / Gestures                         │   │
 * │  │  Foundation                                              │   │
 * │  └─────────────────────────────────────────────────────────┘   │
 * │                           │                                     │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │                   Engine (C++)                           │   │
 * │  │  Skia (2D 渲染) │ Dart Runtime │ Text Rendering          │   │
 * │  └─────────────────────────────────────────────────────────┘   │
 * │                           │                                     │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │                   Embedder                               │   │
 * │  │  Android │ iOS │ Web │ Desktop                           │   │
 * │  └─────────────────────────────────────────────────────────┘   │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * 特点：
 * - 自绘引擎，不依赖原生组件
 * - 一致性好
 * - 性能高（60fps）
 */

// Flutter 组件示例
const flutterExample = `
import 'package:flutter/material.dart';

class MyButton extends StatelessWidget {
  final String title;
  final VoidCallback onPressed;

  const MyButton({
    Key? key,
    required this.title,
    required this.onPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        primary: Colors.blue,
        padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      child: Text(
        title,
        style: TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

// 使用
MyButton(
  title: 'Click Me',
  onPressed: () => print('Clicked!'),
)
`;

// ============================================================
// 4. 小程序原理
// ============================================================

/**
 * 📊 小程序架构
 *
 * 双线程模型：
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        小程序架构                               │
 * │                                                                 │
 * │  ┌──────────────────────┐     ┌──────────────────────┐         │
 * │  │    渲染层            │     │     逻辑层            │         │
 * │  │   (WebView)         │     │    (JS Core)         │         │
 * │  │                      │     │                      │         │
 * │  │  WXML + WXSS        │     │  JavaScript          │         │
 * │  │  渲染页面            │     │  业务逻辑             │         │
 * │  └──────────┬───────────┘     └──────────┬───────────┘         │
 * │             │                            │                      │
 * │             └────────────┬───────────────┘                      │
 * │                          │                                      │
 * │                    ┌─────▼─────┐                                │
 * │                    │  Native   │                                │
 * │                    │  微信客户端│                                │
 * │                    └───────────┘                                │
 * │                                                                 │
 * │  通信方式：setData (JSON 序列化，有性能开销)                      │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * 为什么双线程？
 * - 安全：JS 无法直接操作 DOM
 * - 稳定：JS 卡死不影响渲染
 */

// 小程序组件示例
const miniProgramExample = `
// index.wxml
<view class="container">
  <text>{{message}}</text>
  <button bindtap="handleTap">Click</button>

  <!-- 列表渲染 -->
  <view wx:for="{{list}}" wx:key="id">
    {{item.name}}
  </view>

  <!-- 条件渲染 -->
  <view wx:if="{{show}}">Visible</view>
</view>

// index.js
Page({
  data: {
    message: 'Hello',
    list: [],
    show: true,
  },

  onLoad() {
    this.fetchData();
  },

  async fetchData() {
    const res = await wx.request({ url: '/api/list' });
    this.setData({ list: res.data });
  },

  handleTap() {
    this.setData({ message: 'Clicked!' });
  },
});

// index.wxss
.container {
  padding: 20rpx;
}
`;

// setData 性能优化
const setDataOptimization = `
// ❌ 不好：频繁 setData
for (let i = 0; i < 100; i++) {
  this.setData({ count: i });
}

// ✅ 好：合并更新
this.setData({
  'list[0].name': 'New Name',  // 路径更新
  'obj.a.b': value,            // 深层路径
});

// ✅ 好：批量更新
const updates = {};
for (let i = 0; i < 100; i++) {
  updates[\`list[\${i}].checked\`] = true;
}
this.setData(updates);

// 性能建议：
// 1. 减少 setData 调用次数
// 2. 只更新必要的数据
// 3. 使用路径更新而非全量
// 4. 避免传输大数据
`;

// ============================================================
// 5. Taro/uni-app 多端框架
// ============================================================

/**
 * 📊 多端统一框架
 *
 * Taro：京东出品，React/Vue 语法
 * uni-app：DCloud 出品，Vue 语法
 *
 * 原理：编译时转换为各平台代码
 */

const taroExample = `
// Taro 组件（React 语法）
import { View, Text, Button } from '@tarojs/components';
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <View className="counter">
      <Text>Count: {count}</Text>
      <Button onClick={() => setCount(count + 1)}>+1</Button>
    </View>
  );
}

// 编译为小程序
// <view class="counter">
//   <text>Count: {{count}}</text>
//   <button bindtap="handleClick">+1</button>
// </view>
`;

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. React Native
 *    - Bridge 性能瓶颈
 *    - 原生组件差异
 *    - 调试困难
 *
 * 2. Flutter
 *    - 包体积大
 *    - 热更新受限
 *    - Dart 学习成本
 *
 * 3. 小程序
 *    - setData 性能问题
 *    - 包大小限制
 *    - API 能力受限
 *
 * 4. 通用问题
 *    - 平台差异处理
 *    - 原生能力调用
 *    - 性能优化
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: RN 新架构解决了什么问题？
 * A:
 *    - JSI：去除 Bridge 的序列化开销
 *    - Fabric：同步渲染，更好的动画
 *    - TurboModule：按需加载，启动更快
 *
 * Q2: Flutter 为什么性能好？
 * A:
 *    - 自绘引擎，不经过原生组件
 *    - Dart 支持 AOT 编译
 *    - Skia 高效渲染
 *
 * Q3: 小程序 setData 为什么慢？
 * A:
 *    - 跨线程通信
 *    - JSON 序列化
 *    - 数据量大时开销大
 *
 * Q4: 如何选择跨端方案？
 * A:
 *    - 性能要求高：Flutter
 *    - 团队有 React 经验：RN
 *    - 快速开发小程序：原生/uni-app
 *    - 热更新需求：RN
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：电商 App 技术选型
 */

const techSelectionExample = `
// 需求分析
- 支持 iOS、Android、小程序
- 性能要求：列表流畅、动画丝滑
- 需要热更新
- 团队：前端为主，有 React 经验

// 选型决策
┌─────────────────────────────────────────────────────────────────┐
│ 方案          │ 优势                    │ 劣势                   │
├─────────────────────────────────────────────────────────────────┤
│ RN           │ 热更新、团队熟悉         │ 复杂动画性能一般       │
│ Flutter      │ 性能好、一致性高         │ 不支持热更新           │
│ uni-app      │ 多端覆盖、开发快         │ 性能一般               │
└─────────────────────────────────────────────────────────────────┘

// 最终方案
- 主 App：React Native（热更新 + 团队技能匹配）
- 小程序：Taro（代码复用）
- 核心体验页面：原生开发
`;

export {
  rnComponentExample,
  nativeModuleExample,
  flutterExample,
  miniProgramExample,
  setDataOptimization,
  taroExample,
  techSelectionExample,
};

