"""
评测数据集

管理评测用例
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class TestCase:
    """测试用例"""
    id: str
    input: str
    expected_output: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 用于 RAG 评测
    expected_contexts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "expected_output": self.expected_output,
            "context": self.context,
            "metadata": self.metadata,
            "expected_contexts": self.expected_contexts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        return cls(
            id=data["id"],
            input=data["input"],
            expected_output=data.get("expected_output"),
            context=data.get("context"),
            metadata=data.get("metadata", {}),
            expected_contexts=data.get("expected_contexts", []),
        )


class EvaluationDataset:
    """
    评测数据集
    
    管理测试用例集合
    
    Usage:
        # 创建数据集
        dataset = EvaluationDataset("qa_test", "问答评测")
        dataset.add(TestCase(id="1", input="什么是 RAG？", expected_output="..."))
        
        # 从文件加载
        dataset = EvaluationDataset.load("./eval_data.json")
        
        # 迭代用例
        for case in dataset:
            result = model(case.input)
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._cases: List[TestCase] = []

    def add(self, case: TestCase):
        """添加测试用例"""
        self._cases.append(case)

    def add_many(self, cases: List[TestCase]):
        """批量添加测试用例"""
        self._cases.extend(cases)

    def get(self, case_id: str) -> Optional[TestCase]:
        """根据 ID 获取用例"""
        for case in self._cases:
            if case.id == case_id:
                return case
        return None

    def filter(self, **kwargs: Any) -> "EvaluationDataset":
        """
        过滤用例
        
        Args:
            **kwargs: 过滤条件 (metadata 字段)
        
        Returns:
            新的数据集
        """
        filtered = EvaluationDataset(f"{self.name}_filtered", self.description)
        
        for case in self._cases:
            match = True
            for key, value in kwargs.items():
                if case.metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered.add(case)
        
        return filtered

    def sample(self, n: int, seed: Optional[int] = None) -> "EvaluationDataset":
        """
        随机采样
        
        Args:
            n: 采样数量
            seed: 随机种子
        
        Returns:
            新的数据集
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        sampled = EvaluationDataset(f"{self.name}_sampled", self.description)
        sampled.add_many(random.sample(self._cases, min(n, len(self._cases))))
        
        return sampled

    def save(self, path: str | Path):
        """
        保存数据集
        
        Args:
            path: 文件路径
        """
        path = Path(path)
        
        data = {
            "name": self.name,
            "description": self.description,
            "cases": [case.to_dict() for case in self._cases],
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info("dataset_saved", path=str(path), count=len(self._cases))

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationDataset":
        """
        加载数据集
        
        Args:
            path: 文件路径
        
        Returns:
            EvaluationDataset
        """
        path = Path(path)
        
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        dataset = cls(data["name"], data.get("description", ""))
        
        for case_data in data["cases"]:
            dataset.add(TestCase.from_dict(case_data))
        
        logger.info("dataset_loaded", path=str(path), count=len(dataset))
        
        return dataset

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[TestCase]:
        return iter(self._cases)

    def __getitem__(self, index: int) -> TestCase:
        return self._cases[index]


def create_sample_dataset() -> EvaluationDataset:
    """
    创建示例评测数据集
    
    Returns:
        EvaluationDataset
    """
    dataset = EvaluationDataset(
        "sample_qa",
        "示例问答评测数据集",
    )
    
    # 添加示例用例
    cases = [
        TestCase(
            id="qa_001",
            input="什么是 RAG？",
            expected_output="RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的 AI 技术。",
            metadata={"category": "definition", "difficulty": "easy"},
        ),
        TestCase(
            id="qa_002",
            input="RAG 的主要优势是什么？",
            expected_output="RAG 的主要优势包括：减少幻觉、知识可更新、来源可追溯。",
            metadata={"category": "analysis", "difficulty": "medium"},
        ),
        TestCase(
            id="qa_003",
            input="如何优化 RAG 系统的检索效果？",
            expected_output="优化 RAG 检索效果的方法包括：改进分块策略、使用混合检索、添加重排序步骤、优化嵌入模型。",
            metadata={"category": "how-to", "difficulty": "hard"},
        ),
        TestCase(
            id="qa_004",
            input="Python 中如何实现向量搜索？",
            expected_output="Python 中可以使用 numpy 计算余弦相似度，或使用专门的向量数据库如 ChromaDB、Pinecone。",
            metadata={"category": "implementation", "difficulty": "medium"},
        ),
        TestCase(
            id="qa_005",
            input="FastAPI 如何实现流式响应？",
            expected_output="FastAPI 可以通过 StreamingResponse 或 SSE (Server-Sent Events) 实现流式响应。",
            metadata={"category": "implementation", "difficulty": "medium"},
        ),
    ]
    
    dataset.add_many(cases)
    
    return dataset


