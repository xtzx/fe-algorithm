"""
RAG 系统演示

展示如何构建一个简单的 RAG 问答系统
"""

from pathlib import Path
from llm_kit.rag import (
    DocumentLoader,
    Chunker,
    Embedder,
    VectorIndex,
    Retriever,
)
from llm_kit.prompts import RAGPromptTemplate


def create_sample_docs():
    """创建示例文档"""
    docs_dir = Path("./sample_docs")
    docs_dir.mkdir(exist_ok=True)
    
    # 创建一些示例文档
    (docs_dir / "python.md").write_text("""
# Python 编程语言

Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。

## 特点

- 简洁易读的语法
- 动态类型
- 自动内存管理
- 丰富的标准库

## 应用领域

Python 广泛应用于：
- Web 开发（Django, FastAPI）
- 数据科学和机器学习
- 自动化脚本
- 人工智能
""")

    (docs_dir / "rag.md").write_text("""
# RAG 技术介绍

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。

## 工作原理

1. 检索：根据用户问题，从知识库检索相关文档
2. 增强：将检索到的文档作为上下文
3. 生成：LLM 基于上下文生成答案

## 优势

- 减少幻觉
- 支持最新知识
- 可追溯来源
- 适用于领域知识

## 组件

- 文档加载器（Loader）
- 分块器（Chunker）
- 嵌入器（Embedder）
- 向量索引（Index）
- 检索器（Retriever）
""")
    
    return docs_dir


def main():
    """主函数"""
    print("=== RAG 演示 ===\n")
    
    # 1. 创建示例文档
    print("1. 创建示例文档...")
    docs_dir = create_sample_docs()
    
    # 2. 加载文档
    print("2. 加载文档...")
    loader = DocumentLoader()
    docs = loader.load_directory(docs_dir, pattern="*.md")
    print(f"   加载了 {len(docs)} 个文档")
    
    # 3. 分块
    print("3. 分块处理...")
    chunker = Chunker(chunk_size=300, overlap=50)
    chunks = chunker.split_documents(docs)
    print(f"   生成了 {len(chunks)} 个块")
    
    # 4. 创建索引
    print("4. 创建向量索引...")
    embedder = Embedder(dimension=384)  # Stub 实现
    index = VectorIndex(embedder)
    index.add_chunks(chunks)
    print(f"   索引了 {len(index)} 个向量")
    
    # 5. 创建检索器
    retriever = Retriever(index, top_k=3)
    
    # 6. 测试检索
    print("\n5. 测试检索...")
    
    questions = [
        "什么是 Python？",
        "RAG 有什么优势？",
        "Python 可以用来做什么？",
    ]
    
    template = RAGPromptTemplate()
    
    for q in questions:
        print(f"\n问题: {q}")
        print("-" * 40)
        
        # 检索
        results, citations = retriever.search_with_citations(q)
        
        print(f"找到 {len(results)} 个相关块:")
        for i, result in enumerate(results, 1):
            print(f"  [{i}] 相似度: {result.score:.3f}")
            print(f"      来源: {result.source}")
            print(f"      内容: {result.content[:80]}...")
        
        # 构建提示（这里只展示，不实际调用 LLM）
        context = [
            {"content": r.content, "source": r.source}
            for r in results
        ]
        prompt = template.render(question=q, context=context)
        
        print(f"\n  引用:")
        print(f"  {retriever.format_citations(citations)}")
    
    # 7. 保存索引
    print("\n6. 保存索引...")
    index.save("./rag_index")
    print("   索引已保存到 ./rag_index")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()


