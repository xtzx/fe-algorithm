/**
 * ============================================================
 * 📚 渲染性能优化
 * ============================================================
 *
 * 面试考察重点：
 * 1. 浏览器渲染原理与优化
 * 2. 回流重绘的触发与避免
 * 3. 合成层与 GPU 加速
 * 4. 动画性能优化
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 渲染性能优化的目标
 *
 * 目标：保持 60fps 流畅度，每帧 ≤ 16.67ms
 *
 * 一帧的工作：
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                           16.67ms                                      │
 * ├────────┬────────┬────────┬────────┬────────┬────────┬────────────────┤
 * │   JS   │  Style │ Layout │  Paint │Composite│  Idle  │                │
 * │        │ Recalc │        │        │         │        │                │
 * └────────┴────────┴────────┴────────┴────────┴────────┴────────────────┘
 *
 * 优化目标：尽量减少每一步的耗时
 */

// ============================================================
// 2. 回流与重绘
// ============================================================

/**
 * 📊 回流（Reflow/Layout）
 *
 * 【定义】计算元素的几何属性（位置、大小）
 *
 * 【触发条件】
 * - 添加/删除可见 DOM
 * - 元素位置/尺寸变化
 * - 内容变化（文本、图片尺寸）
 * - 页面初次渲染
 * - 浏览器窗口大小改变
 * - 读取布局属性（强制同步布局）
 *
 * 【会触发回流的属性】
 * - width、height、padding、margin、border
 * - top、left、right、bottom、position
 * - display、float、overflow
 * - font-size、line-height、text-align
 */

/**
 * 📊 重绘（Repaint）
 *
 * 【定义】绘制元素外观（颜色、背景、边框样式等）
 *
 * 【触发条件】
 * - 元素外观变化（不影响布局）
 *
 * 【只触发重绘的属性】
 * - color、background、background-image
 * - border-color、border-style、outline
 * - visibility、box-shadow
 *
 * ⚠️ 关键：回流必定触发重绘，重绘不一定触发回流
 */

/**
 * 📊 强制同步布局（Forced Synchronous Layout）
 *
 * 【问题】读取布局属性时，浏览器必须先完成回流
 *
 * 【触发属性】
 * - offsetTop/Left/Width/Height
 * - scrollTop/Left/Width/Height
 * - clientTop/Left/Width/Height
 * - getComputedStyle()
 * - getBoundingClientRect()
 *
 * ⚠️ 易错点：
 * - 循环中读取布局属性会导致"布局抖动"
 * - 每次读取都会强制回流
 */

// 错误示例：布局抖动
function layoutThrashing() {
  const items = document.querySelectorAll('.item');

  // ❌ 每次循环都会强制回流
  items.forEach(item => {
    const width = item.offsetWidth; // 读取触发回流
    (item as HTMLElement).style.width = width + 10 + 'px'; // 写入
  });
}

// 正确示例：批量读取，批量写入
function optimizedLayout() {
  const items = document.querySelectorAll('.item');

  // ✅ 先批量读取
  const widths = Array.from(items).map(item => item.offsetWidth);

  // ✅ 再批量写入
  items.forEach((item, i) => {
    (item as HTMLElement).style.width = widths[i] + 10 + 'px';
  });
}

// ============================================================
// 3. 合成层与 GPU 加速
// ============================================================

/**
 * 📊 渲染层（Layer）概念
 *
 * 浏览器渲染：
 * 1. 构建 DOM 树
 * 2. 构建渲染树
 * 3. 布局
 * 4. 创建图层树（Layer Tree）
 * 5. 绘制每个图层
 * 6. 合成（Composite）
 *
 * 【普通图层 vs 合成层】
 * - 普通图层：在主线程绑定和合成
 * - 合成层：独立于主线程，GPU 加速
 */

/**
 * 📊 创建合成层的条件
 *
 * 1. transform: translate3d() / translateZ() / scale3d()
 * 2. will-change: transform / opacity
 * 3. opacity 动画（< 1 时）
 * 4. position: fixed
 * 5. video、canvas、iframe 等
 * 6. CSS filter
 *
 * 💡 追问：为什么 transform 不触发回流？
 * A: transform 在合成层处理，不影响文档流和其他元素
 */

/**
 * 📊 GPU 加速的优势
 *
 * 1. 不占用主线程
 * 2. 不触发回流重绘
 * 3. 利用 GPU 并行计算
 *
 * ⚠️ 注意事项：
 * - 合成层占用内存
 * - 过多合成层反而降低性能
 * - 隐式合成可能导致层爆炸
 */

// will-change 使用示例
const willChangeExample = `
/* ✅ 正确用法：悬停时添加 */
.element {
  transition: transform 0.3s;
}
.element:hover {
  will-change: transform;
}
.element:active {
  transform: scale(1.1);
}

/* ❌ 错误用法：全局添加 */
* {
  will-change: transform; /* 内存爆炸！ */
}

/* ✅ JS 动态控制 */
element.addEventListener('mouseenter', () => {
  element.style.willChange = 'transform';
});
element.addEventListener('animationend', () => {
  element.style.willChange = 'auto';
});
`;

// ============================================================
// 4. 动画性能优化
// ============================================================

/**
 * 📊 CSS 动画 vs JS 动画
 *
 * CSS 动画（transform/opacity）：
 * - 不触发回流重绘
 * - 在合成线程执行
 * - 即使主线程繁忙也流畅
 *
 * JS 动画：
 * - 需要手动优化
 * - 主线程执行
 * - 主线程繁忙时会卡顿
 */

/**
 * 📊 高性能动画属性
 *
 * ✅ 只触发合成（最快）：
 * - transform
 * - opacity
 *
 * ⚠️ 触发重绘：
 * - color、background、box-shadow
 *
 * ❌ 触发回流（最慢）：
 * - width、height、margin、padding
 * - top、left、right、bottom
 */

// requestAnimationFrame 动画
function smoothAnimation() {
  const element = document.getElementById('box')!;
  let position = 0;

  function animate() {
    position += 2;
    // ✅ 使用 transform 而不是 left
    element.style.transform = `translateX(${position}px)`;

    if (position < 500) {
      requestAnimationFrame(animate);
    }
  }

  requestAnimationFrame(animate);
}

// FLIP 动画技术
/**
 * 📊 FLIP = First, Last, Invert, Play
 *
 * 原理：
 * 1. First：记录初始位置
 * 2. Last：直接设置到最终位置
 * 3. Invert：计算差值，用 transform 反向偏移
 * 4. Play：移除 transform，让元素"动"到最终位置
 *
 * 优势：使用 transform 动画，性能好
 */
function flipAnimation(element: HTMLElement, finalPosition: DOMRect) {
  // 1. First - 记录初始位置
  const first = element.getBoundingClientRect();

  // 2. Last - 设置最终位置（这里假设已经设置好）
  // element.classList.add('final');
  const last = finalPosition;

  // 3. Invert - 计算差值
  const deltaX = first.left - last.left;
  const deltaY = first.top - last.top;
  const deltaW = first.width / last.width;
  const deltaH = first.height / last.height;

  // 应用反向 transform
  element.style.transform = `translate(${deltaX}px, ${deltaY}px) scale(${deltaW}, ${deltaH})`;
  element.style.transformOrigin = 'top left';

  // 4. Play - 移除 transform，触发动画
  requestAnimationFrame(() => {
    element.style.transition = 'transform 0.3s ease';
    element.style.transform = '';
  });
}

// ============================================================
// 5. 虚拟滚动
// ============================================================

/**
 * 📊 虚拟滚动原理
 *
 * 问题：大量 DOM 节点导致卡顿
 * 解决：只渲染可视区域的元素
 *
 * 原理：
 * ┌────────────────────────────┐
 * │      buffer (上方缓冲区)    │ ← 滚动时提前渲染
 * ├────────────────────────────┤
 * │                            │
 * │      visible (可视区域)     │ ← 实际渲染的 DOM
 * │                            │
 * ├────────────────────────────┤
 * │      buffer (下方缓冲区)    │ ← 滚动时提前渲染
 * └────────────────────────────┘
 */

// 简单虚拟滚动实现
class VirtualList {
  private container: HTMLElement;
  private itemHeight: number;
  private items: any[];
  private visibleCount: number;
  private bufferSize: number;

  constructor(container: HTMLElement, items: any[], itemHeight: number) {
    this.container = container;
    this.items = items;
    this.itemHeight = itemHeight;
    this.visibleCount = Math.ceil(container.clientHeight / itemHeight);
    this.bufferSize = 5; // 上下缓冲 5 个元素

    this.init();
  }

  private init() {
    // 创建占位元素，撑起滚动高度
    const totalHeight = this.items.length * this.itemHeight;
    const placeholder = document.createElement('div');
    placeholder.style.height = `${totalHeight}px`;
    this.container.appendChild(placeholder);

    // 创建内容容器
    const content = document.createElement('div');
    content.style.position = 'absolute';
    content.style.top = '0';
    content.style.left = '0';
    content.style.right = '0';
    this.container.appendChild(content);
    this.container.style.position = 'relative';
    this.container.style.overflow = 'auto';

    // 监听滚动
    this.container.addEventListener('scroll', () => this.onScroll(content));

    // 初始渲染
    this.render(content, 0);
  }

  private onScroll(content: HTMLElement) {
    const scrollTop = this.container.scrollTop;
    const startIndex = Math.max(0, Math.floor(scrollTop / this.itemHeight) - this.bufferSize);
    this.render(content, startIndex);
  }

  private render(content: HTMLElement, startIndex: number) {
    const endIndex = Math.min(
      this.items.length,
      startIndex + this.visibleCount + this.bufferSize * 2
    );

    // 清空并重新渲染
    content.innerHTML = '';
    content.style.transform = `translateY(${startIndex * this.itemHeight}px)`;

    for (let i = startIndex; i < endIndex; i++) {
      const item = document.createElement('div');
      item.style.height = `${this.itemHeight}px`;
      item.textContent = this.items[i];
      content.appendChild(item);
    }
  }
}

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 滥用 will-change
 *    - 会创建合成层，占用内存
 *    - 应该动态添加/移除
 *
 * 2. 不知道哪些属性触发回流
 *    - 导致动画卡顿
 *    - 应该使用 transform/opacity
 *
 * 3. 忽略布局抖动
 *    - 循环中交替读写布局属性
 *    - 应该批量读取，批量写入
 *
 * 4. 大列表不使用虚拟滚动
 *    - 千级 DOM 导致明显卡顿
 *    - 应该使用虚拟滚动
 *
 * 5. 不了解 GPU 加速的代价
 *    - 过多合成层消耗内存
 *    - 移动端尤其明显
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 如何排查渲染性能问题？
 * A:
 * 1. Chrome DevTools Performance 面板
 *    - 查看帧率
 *    - 查看长任务
 *    - 查看渲染时间线
 *
 * 2. Rendering 面板
 *    - Paint flashing：查看重绘区域
 *    - Layer borders：查看图层边界
 *    - FPS meter：实时帧率
 *
 * 3. Layers 面板
 *    - 查看合成层
 *    - 分析层爆炸问题
 *
 * Q2: 什么是层爆炸？如何避免？
 * A:
 * - 过多合成层消耗大量内存
 * - 原因：隐式提升（z-index 覆盖）
 * - 解决：
 *   - 减少 will-change 使用
 *   - 避免动画元素覆盖其他元素
 *   - 使用 contain: paint 隔离
 *
 * Q3: requestAnimationFrame 和 setTimeout 的区别？
 * A:
 * - RAF 与屏幕刷新同步（60fps = 16.67ms）
 * - setTimeout 时间不精确，可能丢帧
 * - RAF 页面不可见时暂停
 * - RAF 在渲染前执行，时机更好
 *
 * Q4: 如何实现 60fps 动画？
 * A:
 * 1. 使用 transform/opacity
 * 2. 使用 requestAnimationFrame
 * 3. 避免布局抖动
 * 4. 减少 DOM 操作
 * 5. 使用 CSS 动画替代 JS 动画
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：列表页滚动卡顿
 *
 * 问题：1000+ 列表项，滚动卡顿
 *
 * 分析：
 * - DOM 节点过多
 * - 每次滚动触发大量回流
 *
 * 解决：
 * 1. 虚拟滚动：只渲染可见区域
 * 2. 使用 transform 替代 top
 * 3. 防抖滚动事件
 *
 * 结果：滚动帧率从 30fps 提升到 60fps
 */

/**
 * 🏢 场景 2：复杂动画卡顿
 *
 * 问题：多元素同时动画，卡顿明显
 *
 * 分析：
 * - 使用 left/top 做动画
 * - 触发大量回流
 *
 * 解决：
 * 1. 改用 transform
 * 2. 使用 will-change 提示
 * 3. 使用 CSS 动画
 *
 * 结果：CPU 占用从 100% 降到 20%
 */

/**
 * 🏢 场景 3：大表格渲染慢
 *
 * 问题：10000 行表格，初始渲染 3s+
 *
 * 分析：
 * - 一次性创建大量 DOM
 * - 阻塞主线程
 *
 * 解决：
 * 1. 虚拟滚动
 * 2. 分批渲染（requestIdleCallback）
 * 3. Web Worker 处理数据
 *
 * 结果：首屏 200ms 内可交互
 */

export {
  layoutThrashing,
  optimizedLayout,
  willChangeExample,
  smoothAnimation,
  flipAnimation,
  VirtualList,
};

