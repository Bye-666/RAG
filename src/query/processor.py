"""Query 处理器

处理用户查询，协调检索流程。
"""
from typing import Any, Dict, List, Optional


class QueryProcessor:
    """查询处理器

    负责查询预处理、向量化、检索协调。
    """

    def __init__(
        self,
        dense_embedder,
        sparse_encoder,
        vector_store,
        reranker=None
    ):
        """初始化处理器

        Args:
            dense_embedder: 稠密向量编码器
            sparse_encoder: 稀疏向量编码器
            vector_store: 向量存储
            reranker: 重排序器（可选）
        """
        self.dense_embedder = dense_embedder
        self.sparse_encoder = sparse_encoder
        self.vector_store = vector_store
        self.reranker = reranker

    def process_query(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_k: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """处理查询

        Args:
            query: 查询文本
            top_k: 初始检索数量
            rerank_top_k: 重排序后返回数量
            **kwargs: 其他参数

        Returns:
            检索结果列表
        """
        # 1. 向量化查询
        dense_vector = self.dense_embedder.encode(query)
        sparse_vector = self.sparse_encoder.encode(query)

        # 2. 混合检索（使用 RRF 融合，由 vector_store 或单独模块实现）
        results = self._hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k
        )

        # 3. 重排序（如果配置）
        if self.reranker and len(results) > 0:
            results = self.reranker.rerank(
                query=query,
                documents=results,
                top_k=rerank_top_k
            )

        return results

    def _hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """混合检索（简化版）

        Args:
            dense_vector: 稠密向量
            sparse_vector: 稀疏向量
            top_k: 返回数量

        Returns:
            检索结果
        """
        # 简化实现：先 dense 检索
        # 生产环境应实现完整的 RRF 融合
        results = self.vector_store.search_dense(
            query_vector=dense_vector,
            top_k=top_k
        )
        return results
