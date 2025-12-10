# 🔬 RAG 进阶

> Rerank、Hybrid Search、GraphRAG、Agentic RAG

---

## Rerank（重排序）

### 原理

```
两阶段检索：
1. 召回阶段：向量检索快速召回 Top-K（如 50 条）
2. 精排阶段：Reranker 对召回结果重新排序

为什么需要 Rerank？
- Embedding 是 Bi-Encoder，编码时不考虑 query-doc 交互
- Reranker 是 Cross-Encoder，同时看 query 和 doc，更精确
- 但 Cross-Encoder 慢，所以分两阶段
```

### 实现

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """重排序"""
        # 构造 query-doc 对
        pairs = [[query, doc] for doc in documents]

        # 计算分数
        scores = self.model.predict(pairs)

        # 排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {"content": doc, "score": float(score)}
            for doc, score in doc_scores[:top_k]
        ]

# 使用 Cohere Rerank API
import cohere

co = cohere.Client("your-api-key")

def cohere_rerank(query: str, documents: List[str], top_k: int = 5):
    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_k,
        model="rerank-english-v2.0"
    )
    return [
        {"content": documents[r.index], "score": r.relevance_score}
        for r in response.results
    ]


# 集成到 RAG Pipeline
class RAGWithRerank:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore()
        self.reranker = Reranker()

    def query(self, question: str, retrieve_k: int = 20, rerank_k: int = 5):
        # 1. 向量检索召回
        query_emb = self.embedder.embed_query(question)
        candidates = self.vector_store.search(query_emb, top_k=retrieve_k)

        # 2. Rerank 精排
        docs = [c['content'] for c in candidates]
        reranked = self.reranker.rerank(question, docs, top_k=rerank_k)

        # 3. 生成答案
        contexts = [r['content'] for r in reranked]
        answer = generate_answer(question, contexts)

        return {"answer": answer, "sources": reranked}
```

---

## Hybrid Search（混合检索）

### 原理

```
结合两种检索方式的优势：
1. 稀疏检索（BM25）：基于关键词匹配，擅长精确匹配
2. 稠密检索（Embedding）：基于语义相似度，擅长理解意图

融合策略：
- 线性加权：score = α * dense_score + (1-α) * sparse_score
- RRF（Reciprocal Rank Fusion）：基于排名融合
```

### 实现

```python
from rank_bm25 import BM25Okapi
import jieba

class HybridSearch:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore()
        self.bm25 = None
        self.documents = []

    def add_documents(self, documents: List[str]):
        """添加文档"""
        self.documents = documents

        # 构建向量索引
        embeddings = self.embedder.embed(documents)
        metadata = [{"content": doc, "index": i} for i, doc in enumerate(documents)]
        self.vector_store.add(embeddings, metadata)

        # 构建 BM25 索引
        tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict]:
        """混合检索"""
        # 1. 向量检索
        query_emb = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_emb, top_k=top_k * 2)

        # 2. BM25 检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_k = np.argsort(bm25_scores)[-top_k * 2:][::-1]

        # 3. 归一化分数
        dense_scores = {r['index']: r['score'] for r in dense_results}
        max_dense = max(dense_scores.values()) if dense_scores else 1

        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        sparse_scores = {i: bm25_scores[i] / max_bm25 for i in bm25_top_k}

        # 4. 融合
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())

        hybrid_scores = []
        for idx in all_indices:
            dense = dense_scores.get(idx, 0) / max_dense
            sparse = sparse_scores.get(idx, 0)
            combined = alpha * dense + (1 - alpha) * sparse
            hybrid_scores.append({
                "index": idx,
                "content": self.documents[idx],
                "score": combined,
                "dense_score": dense,
                "sparse_score": sparse
            })

        # 排序
        hybrid_scores.sort(key=lambda x: x['score'], reverse=True)

        return hybrid_scores[:top_k]


# RRF 融合
def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    """
    Reciprocal Rank Fusion
    rankings: 多个排名列表，每个列表包含文档索引
    """
    scores = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)

    # 按分数排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]
```

---

## GraphRAG

### 原理

```
GraphRAG 使用知识图谱增强检索：
1. 从文档中提取实体和关系
2. 构建知识图谱
3. 检索时同时查询向量和图谱
4. 利用图结构发现隐含关联

优势：
- 捕获实体间关系
- 支持多跳推理
- 更好的上下文理解
```

### 简化实现

```python
import networkx as nx
from typing import Tuple

class SimpleGraphRAG:
    def __init__(self):
        self.graph = nx.Graph()
        self.embedder = EmbeddingModel()
        self.entity_embeddings = {}

    def extract_entities_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """使用 LLM 提取实体和关系"""
        prompt = f"""
从以下文本中提取实体和关系，以 JSON 格式输出。

文本：{text}

输出格式：
{{
    "entities": ["实体1", "实体2", ...],
    "relations": [
        {{"head": "实体1", "relation": "关系类型", "tail": "实体2"}},
        ...
    ]
}}
"""
        response = chat(prompt)

        try:
            data = json.loads(response)
            triples = [
                (r["head"], r["relation"], r["tail"])
                for r in data.get("relations", [])
            ]
            return triples
        except:
            return []

    def add_document(self, text: str, doc_id: str):
        """添加文档到图谱"""
        # 提取实体关系
        triples = self.extract_entities_relations(text)

        # 添加到图
        for head, relation, tail in triples:
            self.graph.add_node(head, type="entity")
            self.graph.add_node(tail, type="entity")
            self.graph.add_edge(head, tail, relation=relation)

        # 存储文档
        self.graph.add_node(doc_id, type="document", content=text)

        # 链接文档到提及的实体
        for head, _, tail in triples:
            self.graph.add_edge(doc_id, head, relation="mentions")
            self.graph.add_edge(doc_id, tail, relation="mentions")

        # 实体 embedding
        entities = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
        if entities:
            embeddings = self.embedder.embed(entities)
            for entity, emb in zip(entities, embeddings):
                self.entity_embeddings[entity] = emb

    def query(self, question: str, top_k: int = 5) -> Dict:
        """图增强检索"""
        # 1. 从问题中提取实体
        question_entities = self._extract_question_entities(question)

        # 2. 在图中找相关实体（语义 + 图邻居）
        related_entities = set(question_entities)

        for entity in question_entities:
            if entity in self.graph:
                # 添加一跳邻居
                neighbors = list(self.graph.neighbors(entity))
                related_entities.update(neighbors)

        # 3. 找到关联的文档
        relevant_docs = set()
        for entity in related_entities:
            if entity in self.graph:
                for neighbor in self.graph.neighbors(entity):
                    if self.graph.nodes[neighbor].get('type') == 'document':
                        relevant_docs.add(neighbor)

        # 4. 获取文档内容
        contexts = []
        for doc_id in relevant_docs:
            content = self.graph.nodes[doc_id].get('content', '')
            if content:
                contexts.append(content)

        # 5. 生成答案
        if contexts:
            answer = generate_answer(question, contexts[:top_k])
        else:
            answer = "未找到相关信息"

        return {
            "answer": answer,
            "entities": list(related_entities),
            "documents": list(relevant_docs)
        }

    def _extract_question_entities(self, question: str) -> List[str]:
        """从问题中提取实体（简化版：基于已有实体匹配）"""
        found = []
        for entity in self.entity_embeddings.keys():
            if entity.lower() in question.lower():
                found.append(entity)
        return found
```

---

## Agentic RAG

### 原理

```
Agentic RAG 将 RAG 与 Agent 结合：
- 根据问题动态决定检索策略
- 支持多轮检索和推理
- 可以调用多种工具

流程：
1. 分析问题，判断是否需要检索
2. 如需要，决定检索什么
3. 评估检索结果是否足够
4. 不足则继续检索或改变策略
5. 最终生成答案
```

### 实现

```python
from enum import Enum
from typing import Optional

class Action(Enum):
    SEARCH = "search"
    REFINE_QUERY = "refine_query"
    ANSWER = "answer"
    NEED_MORE_INFO = "need_more_info"

class AgenticRAG:
    def __init__(self):
        self.rag = RAGWithRerank()
        self.conversation_history = []
        self.retrieved_docs = []

    def _decide_action(self, question: str, current_context: str) -> Tuple[Action, str]:
        """让 LLM 决定下一步行动"""
        prompt = f"""
你是一个 RAG 助手的决策模块。根据当前情况决定下一步行动。

用户问题：{question}

已检索到的信息：
{current_context if current_context else "（暂无）"}

请分析并决定下一步：
1. 如果已有足够信息回答问题，输出：ACTION: answer
2. 如果需要检索更多信息，输出：ACTION: search | QUERY: <优化后的搜索词>
3. 如果当前检索结果不相关，需要换个角度，输出：ACTION: refine_query | QUERY: <新的搜索词>
4. 如果问题无法回答（超出知识库范围），输出：ACTION: need_more_info | REASON: <原因>

只输出一行，格式如上。
"""
        response = chat(prompt)

        # 解析响应
        if "answer" in response.lower():
            return Action.ANSWER, ""
        elif "search" in response.lower():
            query = response.split("QUERY:")[-1].strip() if "QUERY:" in response else question
            return Action.SEARCH, query
        elif "refine" in response.lower():
            query = response.split("QUERY:")[-1].strip() if "QUERY:" in response else question
            return Action.REFINE_QUERY, query
        else:
            return Action.NEED_MORE_INFO, response

    def query(self, question: str, max_iterations: int = 3) -> Dict:
        """Agentic RAG 查询"""
        self.retrieved_docs = []
        current_context = ""

        for i in range(max_iterations):
            # 决定行动
            action, param = self._decide_action(question, current_context)

            if action == Action.ANSWER:
                # 生成最终答案
                answer = generate_answer(question, [d['content'] for d in self.retrieved_docs])
                return {
                    "answer": answer,
                    "sources": self.retrieved_docs,
                    "iterations": i + 1
                }

            elif action in [Action.SEARCH, Action.REFINE_QUERY]:
                # 执行检索
                results = self.rag.query(param)

                # 去重并添加新文档
                for doc in results['sources']:
                    if doc['content'] not in [d['content'] for d in self.retrieved_docs]:
                        self.retrieved_docs.append(doc)

                # 更新上下文
                current_context = "\n\n".join([d['content'] for d in self.retrieved_docs])

            elif action == Action.NEED_MORE_INFO:
                return {
                    "answer": f"抱歉，我无法完全回答这个问题。{param}",
                    "sources": self.retrieved_docs,
                    "iterations": i + 1
                }

        # 达到最大迭代次数
        answer = generate_answer(question, [d['content'] for d in self.retrieved_docs])
        return {
            "answer": answer,
            "sources": self.retrieved_docs,
            "iterations": max_iterations
        }


# Self-RAG：自我反思的 RAG
class SelfRAG:
    """带有自我评估的 RAG"""

    def query(self, question: str) -> Dict:
        # 1. 检索
        results = self.rag.query(question)

        # 2. 评估检索结果相关性
        relevance = self._evaluate_relevance(question, results['sources'])

        if relevance < 0.5:
            # 重新检索或改写查询
            new_query = self._rewrite_query(question)
            results = self.rag.query(new_query)

        # 3. 生成答案
        answer = generate_answer(question, [s['content'] for s in results['sources']])

        # 4. 评估答案质量
        quality = self._evaluate_answer(question, answer, results['sources'])

        if quality < 0.5:
            # 尝试改进答案
            answer = self._improve_answer(question, answer, results['sources'])

        # 5. 检查幻觉
        is_grounded = self._check_grounding(answer, results['sources'])

        return {
            "answer": answer,
            "sources": results['sources'],
            "confidence": quality,
            "grounded": is_grounded
        }
```

---

## LangChain RAG 实现

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# 1. 加载文档
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. 切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# 3. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)

# 4. 创建 RAG Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# 5. 查询
result = qa_chain.invoke({"query": "文档的主要内容是什么？"})
print(result["result"])
```

---

## 练习题

### 练习 1：实现 Rerank

```python
# 任务：集成 Reranker 到现有 RAG 系统
# 对比有无 Rerank 的效果差异
```

### 练习 2：Hybrid Search

```python
# 任务：实现一个混合检索系统
# 支持调节 BM25 和向量检索的权重
# 测试不同权重对结果的影响
```

### 练习 3：构建 GraphRAG

```python
# 任务：使用 Neo4j 或 NetworkX 构建一个简单的 GraphRAG
# 能够回答需要多跳推理的问题
```

---

## 小结

```
本节要点：
1. Rerank：两阶段检索，Cross-Encoder 精排
2. Hybrid Search：BM25 + 向量，兼顾精确和语义
3. GraphRAG：知识图谱增强，支持关系推理
4. Agentic RAG：动态决策，多轮检索
```

---

## ➡️ 下一步

继续 [05-Agent基础.md](./05-Agent基础.md)

