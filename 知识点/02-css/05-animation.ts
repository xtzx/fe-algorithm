/**
 * ============================================================
 * 📚 CSS 动画与性能
 * ============================================================
 *
 * 面试考察重点：
 * 1. transition 与 animation 的区别
 * 2. CSS 动画性能优化
 * 3. FLIP 动画技术
 * 4. 高性能动画实践
 */

// ============================================================
// 1. Transition（过渡）
// ============================================================

/**
 * 📖 transition 基础
 *
 * 用于在属性值改变时添加过渡效果
 *
 * 语法：transition: property duration timing-function delay;
 */

const transitionBasic = `
  .button {
    background: #1890ff;
    transform: scale(1);

    /* 单个属性 */
    transition: background 0.3s ease;

    /* 多个属性 */
    transition:
      background 0.3s ease,
      transform 0.2s ease-out;

    /* 所有属性 */
    transition: all 0.3s ease;  /* 不推荐：性能差 */
  }

  .button:hover {
    background: #096dd9;
    transform: scale(1.05);
  }
`;

/**
 * 📊 timing-function 缓动函数
 *
 * ease         - 默认，慢-快-慢
 * linear       - 匀速
 * ease-in      - 慢-快
 * ease-out     - 快-慢
 * ease-in-out  - 慢-快-慢（比 ease 更平滑）
 * cubic-bezier - 自定义贝塞尔曲线
 *
 * 📊 贝塞尔曲线示例
 *
 * cubic-bezier(0.68, -0.55, 0.27, 1.55)  // 弹性效果
 * cubic-bezier(0.4, 0, 0.2, 1)           // Material Design 标准
 * cubic-bezier(0.25, 0.1, 0.25, 1)       // 苹果风格
 *
 * 工具：https://cubic-bezier.com/
 */

const timingFunctions = `
  /* 常用缓动效果 */
  .ease-smooth {
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
  }

  .ease-bounce {
    transition-timing-function: cubic-bezier(0.68, -0.55, 0.27, 1.55);
  }

  .ease-elastic {
    transition-timing-function: cubic-bezier(0.68, -0.6, 0.32, 1.6);
  }

  /* steps 阶梯函数 - 逐帧动画 */
  .sprite-animation {
    transition: background-position 0.5s steps(8);
  }
`;

// ============================================================
// 2. Animation（动画）
// ============================================================

/**
 * 📖 animation 基础
 *
 * 用于创建复杂的关键帧动画
 *
 * 语法：animation: name duration timing-function delay iteration-count direction fill-mode play-state;
 */

const animationBasic = `
  /* 定义关键帧 */
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes bounce {
    0%, 100% {
      transform: translateY(0);
    }
    50% {
      transform: translateY(-30px);
    }
  }

  /* 使用动画 */
  .fade-in {
    animation: fadeIn 0.5s ease forwards;
  }

  .bounce {
    animation: bounce 1s ease-in-out infinite;
  }
`;

/**
 * 📊 animation 属性详解
 *
 * animation-name          - 关键帧名称
 * animation-duration      - 持续时间
 * animation-timing-function - 缓动函数
 * animation-delay         - 延迟
 * animation-iteration-count - 重复次数（infinite 无限）
 * animation-direction     - 方向（normal/reverse/alternate/alternate-reverse）
 * animation-fill-mode     - 结束状态（none/forwards/backwards/both）
 * animation-play-state    - 播放状态（running/paused）
 */

const animationAdvanced = `
  /* 多动画组合 */
  .complex-animation {
    animation:
      fadeIn 0.5s ease forwards,
      pulse 2s ease-in-out 0.5s infinite;
  }

  /* 延迟序列动画 */
  .item:nth-child(1) { animation-delay: 0s; }
  .item:nth-child(2) { animation-delay: 0.1s; }
  .item:nth-child(3) { animation-delay: 0.2s; }
  .item:nth-child(4) { animation-delay: 0.3s; }

  /* CSS 变量控制延迟 */
  .item {
    animation: fadeIn 0.5s ease forwards;
    animation-delay: calc(var(--i) * 0.1s);
  }

  /* 悬停暂停 */
  .carousel {
    animation: scroll 20s linear infinite;
  }
  .carousel:hover {
    animation-play-state: paused;
  }
`;

// ============================================================
// 3. Transform（变换）
// ============================================================

/**
 * 📊 transform 属性
 *
 * 2D 变换：
 * - translate(x, y) / translateX / translateY
 * - rotate(angle)
 * - scale(x, y) / scaleX / scaleY
 * - skew(x, y) / skewX / skewY
 *
 * 3D 变换：
 * - translate3d(x, y, z) / translateZ
 * - rotate3d(x, y, z, angle) / rotateX / rotateY / rotateZ
 * - scale3d(x, y, z) / scaleZ
 * - perspective(n)
 */

const transformExamples = `
  /* 2D 变换 */
  .card-2d {
    transform: translateY(-10px) rotate(5deg) scale(1.1);
  }

  /* 3D 变换 */
  .card-3d {
    perspective: 1000px;  /* 在父元素上设置 */
  }
  .card-3d:hover .front {
    transform: rotateY(180deg);
  }

  /* 3D 翻转卡片 */
  .flip-card {
    perspective: 1000px;
  }
  .flip-card-inner {
    position: relative;
    transform-style: preserve-3d;  /* 保持 3D 空间 */
    transition: transform 0.6s;
  }
  .flip-card:hover .flip-card-inner {
    transform: rotateY(180deg);
  }
  .flip-card-front,
  .flip-card-back {
    position: absolute;
    backface-visibility: hidden;  /* 隐藏背面 */
  }
  .flip-card-back {
    transform: rotateY(180deg);
  }

  /* transform-origin 变换原点 */
  .rotate-corner {
    transform-origin: top left;
    transform: rotate(45deg);
  }
`;

// ============================================================
// 4. 动画性能优化
// ============================================================

/**
 * 📊 高性能属性 vs 低性能属性
 *
 * ✅ 高性能（只触发合成）：
 * - transform
 * - opacity
 * - filter（部分）
 *
 * ⚠️ 中性能（触发重绘）：
 * - color
 * - background
 * - box-shadow
 *
 * ❌ 低性能（触发回流）：
 * - width/height
 * - padding/margin
 * - top/left/right/bottom
 * - font-size
 */

const performanceOptimization = `
  /* ❌ 避免：使用位置属性做动画 */
  .bad-animation {
    position: absolute;
    transition: left 0.3s, top 0.3s;
  }
  .bad-animation:hover {
    left: 100px;
    top: 50px;
  }

  /* ✅ 推荐：使用 transform */
  .good-animation {
    transition: transform 0.3s;
  }
  .good-animation:hover {
    transform: translate(100px, 50px);
  }

  /* ❌ 避免：使用 width/height 做动画 */
  .bad-resize {
    transition: width 0.3s, height 0.3s;
  }

  /* ✅ 推荐：使用 scale */
  .good-resize {
    transition: transform 0.3s;
  }
  .good-resize:hover {
    transform: scale(1.5);
  }

  /* will-change 提升性能 */
  .will-animate {
    will-change: transform, opacity;
  }

  /* 动画结束后移除 */
  .animated {
    animation: slideIn 0.5s ease forwards;
  }
  /* JavaScript: element.addEventListener('animationend', () => {
     element.style.willChange = 'auto';
   }); */
`;

// ============================================================
// 5. FLIP 动画技术
// ============================================================

/**
 * 📖 什么是 FLIP？
 *
 * FLIP = First Last Invert Play
 *
 * 一种高性能动画技术，适用于位置/尺寸变化的动画。
 *
 * 📊 FLIP 原理
 *
 * 1. First：记录元素的初始状态（位置、尺寸）
 * 2. Last：记录元素的最终状态
 * 3. Invert：计算差值，用 transform 将元素"反转"到初始位置
 * 4. Play：移除 transform，让元素动画到最终位置
 *
 * 优势：始终使用 transform 做动画，性能最佳
 */

const flipExample = `
  /* FLIP 动画示例 */

  // JavaScript 实现
  function flipAnimate(element, callback) {
    // 1. First - 记录初始状态
    const first = element.getBoundingClientRect();

    // 2. Last - 执行 DOM 变化
    callback();

    // 3. 记录最终状态
    const last = element.getBoundingClientRect();

    // 4. Invert - 计算差值
    const deltaX = first.left - last.left;
    const deltaY = first.top - last.top;
    const deltaW = first.width / last.width;
    const deltaH = first.height / last.height;

    // 5. 应用反转变换
    element.style.transform = \`
      translate(\${deltaX}px, \${deltaY}px)
      scale(\${deltaW}, \${deltaH})
    \`;
    element.style.transformOrigin = 'top left';

    // 强制重绘
    element.offsetHeight;

    // 6. Play - 添加过渡并移除变换
    element.style.transition = 'transform 0.3s ease';
    element.style.transform = '';

    // 清理
    element.addEventListener('transitionend', () => {
      element.style.transition = '';
      element.style.transformOrigin = '';
    }, { once: true });
  }

  // 使用
  // flipAnimate(card, () => {
  //   card.classList.toggle('expanded');
  // });
`;

/**
 * 📊 FLIP 应用场景
 *
 * 1. 列表重排动画
 * 2. 共享元素过渡
 * 3. 布局变化动画
 * 4. 图片展开效果
 *
 * 相关库：
 * - GSAP Flip Plugin
 * - Flipping.js
 * - Vue <transition-group>（内置 FLIP）
 */

// ============================================================
// 6. 常见动画效果
// ============================================================

const commonAnimations = `
  /* 1. 淡入淡出 */
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  /* 2. 缩放 */
  @keyframes scaleIn {
    from {
      opacity: 0;
      transform: scale(0.9);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  @keyframes pulse {
    0%, 100% {
      transform: scale(1);
    }
    50% {
      transform: scale(1.05);
    }
  }

  /* 3. 旋转 */
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* 4. 抖动 */
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-10px); }
    75% { transform: translateX(10px); }
  }

  /* 5. 弹跳 */
  @keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
      transform: translateY(0);
    }
    40% {
      transform: translateY(-30px);
    }
    60% {
      transform: translateY(-15px);
    }
  }

  /* 6. 骨架屏 Shimmer */
  @keyframes shimmer {
    0% {
      background-position: -200% 0;
    }
    100% {
      background-position: 200% 0;
    }
  }
  .skeleton {
    background: linear-gradient(
      90deg,
      #f0f0f0 25%,
      #e0e0e0 50%,
      #f0f0f0 75%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
  }

  /* 7. 加载动画 */
  .spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #1890ff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  /* 8. 打字机效果 */
  @keyframes typing {
    from { width: 0; }
    to { width: 100%; }
  }
  .typewriter {
    overflow: hidden;
    white-space: nowrap;
    border-right: 2px solid;
    animation:
      typing 3s steps(30, end),
      blink 0.5s step-end infinite alternate;
  }
  @keyframes blink {
    50% { border-color: transparent; }
  }
`;

// ============================================================
// 7. 高频面试题
// ============================================================

/**
 * 题目 1：transition 和 animation 的区别？
 *
 * transition：
 * - 需要触发条件（:hover、:focus、class 变化等）
 * - 只有开始和结束两个状态
 * - 只能执行一次
 *
 * animation：
 * - 不需要触发条件，可以自动执行
 * - 可以定义多个关键帧
 * - 可以无限循环
 * - 更多控制（方向、填充模式、暂停等）
 */

/**
 * 题目 2：如何实现 60fps 的流畅动画？
 *
 * 1. 只使用 transform 和 opacity
 * 2. 使用 will-change 提前告知浏览器
 * 3. 使用 requestAnimationFrame（JS 动画）
 * 4. 避免在动画中读取布局属性
 * 5. 减少合成层数量
 * 6. 使用 contain 属性限制影响范围
 */

/**
 * 题目 3：GPU 加速动画的原理？注意事项？
 *
 * 原理：
 * - 将元素提升为独立的合成层
 * - 在 GPU 上独立渲染和合成
 * - 不需要主线程参与
 *
 * 注意事项：
 * - 每个合成层消耗额外内存
 * - 过多合成层反而降低性能
 * - 可能导致字体渲染模糊
 * - 动画结束后移除 will-change
 */

/**
 * 题目 4：CSS 动画卡顿如何排查？
 *
 * 1. Chrome DevTools → Performance 面板
 *    - 查看 FPS、CPU、Main 线程
 *    - 检查是否有长任务
 *
 * 2. Chrome DevTools → Rendering
 *    - Paint flashing：查看重绘区域
 *    - Layer borders：查看合成层
 *    - FPS meter：实时帧率
 *
 * 3. 常见原因：
 *    - 动画属性触发回流
 *    - JS 阻塞主线程
 *    - 过多合成层
 *    - 同时动画元素太多
 */

export {
  transitionBasic,
  timingFunctions,
  animationBasic,
  animationAdvanced,
  transformExamples,
  performanceOptimization,
  flipExample,
  commonAnimations,
};

