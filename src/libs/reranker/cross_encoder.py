"""Cross-Encoder Reranker 实现

使用 sentence-transformers 的 Cross-Encoder 模型进行重排序。
"""
from typing import Any, Dict, List

from .base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder Reranker 实现

    使用 Cross-Encoder 模型计算查询和文档的相关性分数。
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        **kwargs: Any
    ):
        """初始化 Cross-Encoder Reranker

        Args:
            model: Cross-Encoder 模型名称
            **kwargs: 其他参数
        """
        super().__init__(model=model, **kwargs)

        # 延迟导入
        try:
            from sentence_transformers import CrossEncoder
            self.CrossEncoder = CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers 包未安装。请运行：pip install sentence-transformers"
            )

        # 加载模型
        self.encoder = self.CrossEncoder(self.model)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回结果数量
            **kwargs: 其他参数

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        # 限制 top_k
        top_k = min(top_k, len(documents))

        # 构建输入对
        pairs = [[query, doc["text"]] for doc in documents]

        # 计算相关性分数
        scores = self.encoder.predict(pairs)

        # 更新文档分数
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        # 按分数排序
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        # 返回 top_k
        return reranked[:top_k]
