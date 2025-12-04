/**
 * ============================================================
 * 📚 CSS 深入 - 高频面试题汇总
 * ============================================================
 *
 * 按照面试频率和难度进行分类
 */

// ============================================================
// 🔥🔥🔥 高频必考题
// ============================================================

/**
 * 1. 盒模型是什么？标准盒模型和 IE 盒模型的区别？
 *
 * 盒模型：content + padding + border + margin
 *
 * 标准盒模型（content-box）：
 * - width/height 只包含 content
 * - 实际宽度 = width + padding + border
 *
 * IE 盒模型（border-box）：
 * - width/height 包含 content + padding + border
 * - 实际宽度 = width
 *
 * 切换：box-sizing: content-box | border-box
 * 推荐：全局使用 border-box
 *
 * ⚠️ 易错点：
 * - margin 不计入盒模型尺寸，但影响布局
 * - outline 不占用空间，不影响布局
 * - 行内元素的垂直 padding/margin 有特殊表现
 *
 * 💡 面试追问：
 * Q: 为什么推荐 border-box？
 * A: 更直观，设置 width 就是最终宽度，方便计算布局
 */

/**
 * 2. 什么是 BFC？如何创建？有什么用？
 *
 * BFC（块级格式化上下文）：独立的渲染区域，内部布局不影响外部。
 *
 * 创建方式：
 * - float 不为 none
 * - position 为 absolute/fixed
 * - display 为 inline-block/flex/grid/table-cell 等
 * - overflow 不为 visible（推荐 auto）
 * - display: flow-root（最佳方案，无副作用）
 *
 * 作用：
 * - 清除浮动（解决高度塌陷）
 * - 防止 margin 折叠
 * - 阻止元素被浮动元素覆盖
 *
 * ⚠️ 易错点：
 * - overflow: hidden 会裁剪内容，不是所有场景都适用
 * - flex/grid 容器天然是 BFC
 * - BFC 只能包含子元素的浮动，不能包含自身
 *
 * 💡 面试追问：
 * Q: 还有哪些格式化上下文？
 * A: IFC（行内）、GFC（Grid）、FFC（Flex）
 */

/**
 * 3. Flex 布局常用属性？
 *
 * 容器属性：
 * - flex-direction：主轴方向
 * - flex-wrap：是否换行
 * - justify-content：主轴对齐
 * - align-items：交叉轴对齐
 * - gap：间距
 *
 * 项目属性：
 * - flex-grow：放大比例
 * - flex-shrink：缩小比例
 * - flex-basis：初始大小
 * - flex: 1 = flex: 1 1 0%
 *
 * ⚠️ 易错点：
 * - flex: 1 和 flex: auto 不同（0% vs auto）
 * - flex-shrink 默认是 1，元素会收缩
 * - min-width 默认是 auto，可能导致内容溢出
 *
 * 💡 面试追问：
 * Q: flex: 1 具体代表什么？
 * A: flex-grow: 1, flex-shrink: 1, flex-basis: 0%
 *    元素会平分剩余空间
 *
 * Q: 如何让 flex 子元素不收缩？
 * A: flex-shrink: 0 或 min-width: 0
 */

/**
 * 4. 水平垂直居中的方案？
 *
 * 1. Flex（推荐）：
 *    display: flex; justify-content: center; align-items: center;
 *
 * 2. Grid：
 *    display: grid; place-items: center;
 *
 * 3. 绝对定位 + transform：
 *    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
 *
 * 4. 绝对定位 + margin auto（需要固定宽高）：
 *    position: absolute; inset: 0; margin: auto; width: 100px; height: 100px;
 */

/**
 * 5. 移动端适配方案？
 *
 * 1. rem 方案：
 *    - 根据屏幕宽度动态设置 html font-size
 *    - 元素使用 rem 单位
 *    - 需要 JS 计算
 *
 * 2. vw 方案（推荐）：
 *    - 纯 CSS，无需 JS
 *    - 1vw = 视口宽度的 1%
 *    - 使用 postcss-px-to-viewport 转换
 *
 * 3. rem + vw：
 *    - 用 vw 设置根 font-size
 *    - 无需 JS
 */

/**
 * 6. 回流和重绘是什么？如何优化？
 *
 * 回流（Reflow）：
 * - 计算元素几何属性（位置、大小）
 * - 触发：添加/删除元素、尺寸变化、窗口大小变化
 *
 * 重绘（Repaint）：
 * - 绘制元素外观（颜色、背景）
 * - 触发：颜色变化、阴影变化
 *
 * 关系：回流必触发重绘，重绘不一定触发回流
 *
 * 优化：
 * - 使用 transform/opacity 做动画
 * - 批量修改 DOM
 * - 缓存布局信息
 * - 使用 will-change
 */

// ============================================================
// 🔥🔥 进阶深入题
// ============================================================

/**
 * 7. 层叠上下文是什么？z-index 为什么不生效？
 *
 * 层叠上下文：元素的 Z 轴层叠排列
 *
 * 创建条件：
 * - position + z-index（非 auto）
 * - opacity < 1
 * - transform/filter 不为 none
 * - flex/grid 子元素 + z-index
 *
 * z-index 不生效原因：
 * - 元素不是定位元素（static）
 * - 父元素创建了层叠上下文，比较的是父元素的层级
 */

/**
 * 8. CSS 选择器优先级？
 *
 * 优先级从高到低：
 * 1. !important
 * 2. 内联样式（1000）
 * 3. ID 选择器（100）
 * 4. 类/属性/伪类选择器（10）
 * 5. 元素/伪元素选择器（1）
 * 6. 通配符/关系选择器（0）
 *
 * 计算规则：按权重累加，不进位
 */

/**
 * 9. CSS 隐藏元素的方式？区别？
 *
 * display: none
 * - 完全隐藏，不占空间
 * - 不可交互
 * - 触发回流
 *
 * visibility: hidden
 * - 隐藏但占空间
 * - 不可交互
 * - 只触发重绘
 * - 子元素可设置 visible
 *
 * opacity: 0
 * - 透明但占空间
 * - 可交互（点击事件有效）
 * - 只触发合成
 *
 * position: absolute + 移出视口
 * - 不占空间
 * - 不影响布局
 * - 可用于可访问性（屏幕阅读器）
 */

/**
 * 10. 1px 边框问题怎么解决？
 *
 * 问题：Retina 屏幕上 1px CSS = 多个物理像素，边框看起来很粗
 *
 * 解决方案：
 *
 * 1. transform + 伪元素（推荐）
 *    ::after { transform: scale(0.5); }
 *
 * 2. viewport 缩放
 *    根据 devicePixelRatio 设置 initial-scale
 *
 * 3. box-shadow
 *    box-shadow: 0 0 0 0.5px #ccc;
 *
 * 4. svg 或 图片
 */

/**
 * 11. CSS 动画和 JS 动画的区别？什么时候用哪个？
 *
 * CSS 动画：
 * - 适合简单动画
 * - 性能好（可启用 GPU 加速）
 * - 不阻塞主线程
 * - 控制能力有限
 *
 * JS 动画：
 * - 适合复杂动画
 * - 控制精细（暂停、反向、动态参数）
 * - 可能阻塞主线程
 * - 配合 requestAnimationFrame 性能也不错
 *
 * 使用场景：
 * - 简单过渡效果：CSS transition
 * - 循环/关键帧动画：CSS animation
 * - 复杂交互动画：JS + requestAnimationFrame
 * - 物理动画/弹簧效果：JS 动画库（如 GSAP）
 */

/**
 * 12. CSS 变量怎么用？和 SCSS 变量的区别？
 *
 * CSS 变量：
 * ```css
 * :root { --primary: #1890ff; }
 * .btn { color: var(--primary); }
 * ```
 *
 * 区别：
 * - CSS 变量运行时生效，SCSS 变量编译时
 * - CSS 变量可以用 JS 动态修改
 * - CSS 变量可以继承和覆盖
 * - SCSS 变量功能更丰富（计算、循环等）
 */

// ============================================================
// 📝 经典代码题
// ============================================================

/**
 * 题目 1：实现三角形
 */
const triangle = `
  .triangle {
    width: 0;
    height: 0;
    border: 50px solid transparent;
    border-bottom-color: red;
  }

  /* 等边三角形 */
  .equilateral {
    width: 0;
    height: 0;
    border-left: 50px solid transparent;
    border-right: 50px solid transparent;
    border-bottom: 87px solid red;  /* 50 * √3 */
  }
`;

/**
 * 题目 2：实现两栏布局（左固定右自适应）
 */
const twoColumn = `
  /* Flex */
  .container { display: flex; }
  .left { width: 200px; flex-shrink: 0; }
  .right { flex: 1; }

  /* Grid */
  .container {
    display: grid;
    grid-template-columns: 200px 1fr;
  }

  /* 浮动 + BFC */
  .left { float: left; width: 200px; }
  .right { overflow: hidden; }
`;

/**
 * 题目 3：实现三栏布局（两侧固定中间自适应）
 */
const threeColumn = `
  /* Flex */
  .container { display: flex; }
  .left, .right { width: 200px; flex-shrink: 0; }
  .center { flex: 1; }

  /* Grid */
  .container {
    display: grid;
    grid-template-columns: 200px 1fr 200px;
  }
`;

/**
 * 题目 4：实现单行/多行文本溢出省略
 */
const textEllipsis = `
  /* 单行 */
  .single-line {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* 多行 */
  .multi-line {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
`;

/**
 * 题目 5：清除浮动
 */
const clearFloat = `
  /* 推荐：clearfix */
  .clearfix::after {
    content: '';
    display: block;
    clear: both;
  }

  /* 或者：触发 BFC */
  .parent {
    overflow: hidden;
  }
`;

/**
 * 题目 6：实现等高布局
 */
const equalHeight = `
  /* Flex（推荐） */
  .container {
    display: flex;
    /* align-items 默认 stretch */
  }

  /* Grid */
  .container {
    display: grid;
    grid-auto-flow: column;
  }

  /* Table */
  .container { display: table; }
  .item { display: table-cell; }
`;

// ============================================================
// 🎯 资深面试追问清单
// ============================================================

/**
 * 追问 1：为什么 transform 动画性能好？
 *
 * - transform 只触发合成层，不触发回流重绘
 * - 合成层在 GPU 上独立渲染
 * - 动画在合成线程执行，不阻塞主线程
 */

/**
 * 追问 2：will-change 的原理和注意事项？
 *
 * 原理：
 * - 告诉浏览器元素将要变化
 * - 浏览器提前做优化（如创建合成层）
 *
 * 注意：
 * - 不要滥用，有内存开销
 * - 动画结束后移除
 * - 不要放在 :hover 等伪类里
 */

/**
 * 追问 3：CSS 选择器是从右向左匹配的，为什么？
 *
 * 从右向左可以快速排除不匹配的元素：
 * - 先找到所有符合最右边选择器的元素
 * - 再向左逐级验证祖先
 * - 大部分情况在第一步就排除了大量元素
 *
 * 如果从左向右：
 * - 需要遍历所有后代
 * - 可能匹配很多无效路径
 */

/**
 * 追问 4：CSS Modules 和 CSS-in-JS 怎么选择？
 *
 * CSS Modules：
 * - 零运行时
 * - 更接近传统 CSS 写法
 * - 适合大部分场景
 *
 * CSS-in-JS：
 * - 需要动态样式的场景
 * - 需要主题系统的场景
 * - 组件库开发
 *
 * 原子化 CSS（Tailwind）：
 * - 追求极致性能
 * - 样式高度复用的场景
 */

// ============================================================
// 📋 面试准备 Checklist
// ============================================================

/**
 * ✅ 盒模型
 *    - [ ] 标准 vs IE 盒模型
 *    - [ ] box-sizing 属性
 *
 * ✅ 布局
 *    - [ ] BFC 概念和应用
 *    - [ ] Flex 布局详解
 *    - [ ] Grid 布局基础
 *    - [ ] 居中方案
 *    - [ ] 两栏/三栏布局
 *
 * ✅ 定位与层叠
 *    - [ ] position 各值
 *    - [ ] 层叠上下文
 *    - [ ] z-index 规则
 *
 * ✅ 渲染性能
 *    - [ ] 回流与重绘
 *    - [ ] 合成层与 GPU 加速
 *    - [ ] 动画性能优化
 *
 * ✅ 响应式
 *    - [ ] 移动端适配方案
 *    - [ ] 媒体查询
 *    - [ ] 1px 问题
 *
 * ✅ 工程化
 *    - [ ] 样式隔离方案
 *    - [ ] CSS 预处理器
 *    - [ ] CSS 变量与主题
 */

export {
  triangle,
  twoColumn,
  threeColumn,
  textEllipsis,
  clearFloat,
  equalHeight,
};

