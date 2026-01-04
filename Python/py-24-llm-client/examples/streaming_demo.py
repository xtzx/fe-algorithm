"""
流式输出演示

展示如何处理 LLM 的流式响应
"""

import time
from llm_kit.client.streaming import StreamProcessor, StreamCollector, StreamInterrupter
from llm_kit.client.base import StreamChunk


def simulate_stream():
    """模拟流式输出"""
    text = "Python 是一种高级编程语言，以其简洁易读的语法而闻名。它广泛应用于 Web 开发、数据科学、人工智能等领域。"
    
    for char in text:
        yield StreamChunk(delta=char)
        time.sleep(0.02)  # 模拟延迟
    
    yield StreamChunk(delta="", finish_reason="stop")


def demo_basic_streaming():
    """基础流式输出"""
    print("=== 基础流式输出 ===")
    print("输出: ", end="")
    
    for chunk in simulate_stream():
        print(chunk.delta, end="", flush=True)
        if chunk.is_done:
            break
    
    print("\n")


def demo_stream_processor():
    """流式处理器"""
    print("=== 流式处理器 ===")
    
    processor = StreamProcessor()
    print("输出: ", end="")
    
    for chunk in simulate_stream():
        processor.process(chunk)
        print(chunk.delta, end="", flush=True)
    
    result = processor.get_result()
    print(f"\n\n统计:")
    print(f"  总字符: {len(result.content)}")
    print(f"  块数量: {result.chunk_count}")
    print(f"  完成原因: {result.finish_reason}")
    print()


def demo_stream_collector():
    """流式收集器"""
    print("=== 流式收集器 ===")
    
    def on_delta(delta):
        print(delta, end="", flush=True)
    
    print("输出: ", end="")
    
    for chunk in StreamCollector.collect(simulate_stream(), on_delta=on_delta):
        pass
    
    print("\n")


def demo_stream_interrupter():
    """流式中断"""
    print("=== 流式中断 ===")
    
    interrupter = StreamInterrupter()
    char_count = 0
    
    print("输出（10个字符后中断）: ", end="")
    
    for chunk in interrupter.wrap(simulate_stream()):
        print(chunk.delta, end="", flush=True)
        char_count += len(chunk.delta)
        
        # 模拟用户取消
        if char_count >= 10:
            interrupter.interrupt()
    
    print(f"\n\n已中断: {interrupter.was_interrupted}")
    print(f"已收集: {interrupter.content}")
    print()


def main():
    print("=" * 50)
    print("流式输出演示")
    print("=" * 50)
    print()
    
    demo_basic_streaming()
    demo_stream_processor()
    demo_stream_collector()
    demo_stream_interrupter()
    
    print("演示完成！")


if __name__ == "__main__":
    main()


