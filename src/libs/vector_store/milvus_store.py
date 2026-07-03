"""Milvus 向量存储实现

基于 Milvus Lite，支持双向量（Dense + Sparse）混合检索。
"""
import os
from typing import Any, Dict, List, Optional

from .base import BaseVectorStore


class MilvusStore(BaseVectorStore):
    """Milvus 向量存储实现

    支持：
    - Milvus Lite（本地文件模式）
    - 双向量检索（Dense + Sparse）
    - 元数据过滤
    """

    def __init__(
        self,
        uri: str = "./data/db/milvus.db",
        collection_name: str = "rag_collection",
        dense_dim: int = 2048,
        **kwargs: Any
    ):
        """初始化 Milvus 存储

        Args:
            uri: Milvus 连接 URI
                - 本地文件路径：Milvus Lite 模式
                - http://...：标准 Milvus 服务
            collection_name: 集合名称
            dense_dim: 稠密向量维度
            **kwargs: 其他参数
        """
        super().__init__(collection_name=collection_name, **kwargs)
        self.uri = uri
        self.dense_dim = dense_dim

        # 延迟导入
        try:
            from pymilvus import MilvusClient, DataType
            self.MilvusClient = MilvusClient
            self.DataType = DataType
        except ImportError:
            raise ImportError(
                "pymilvus 包未安装。请运行：pip install pymilvus"
            )

        # 连接 Milvus
        self.client = MilvusClient(uri=self.uri)

        # 确保集合存在
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """确保集合存在，如果不存在则创建"""
        # 检查集合是否存在
        if self.client.has_collection(collection_name=self.collection_name):
            return

        # 创建集合 Schema
        schema = self.MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,  # 支持动态字段（metadata）
        )

        # 添加字段
        schema.add_field(
            field_name="id",
            datatype=self.DataType.VARCHAR,
            is_primary=True,
            max_length=512
        )
        schema.add_field(
            field_name="text",
            datatype=self.DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=self.DataType.FLOAT_VECTOR,
            dim=self.dense_dim
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=self.DataType.SPARSE_FLOAT_VECTOR
        )

        # 创建索引参数
        index_params = self.MilvusClient.prepare_index_params()

        # Dense 索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # Sparse 索引
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def upsert(
        self,
        ids: List[str],
        texts: List[str],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> None:
        """插入或更新向量

        Args:
            ids: 文档 ID 列表
            texts: 文本内容列表
            dense_vectors: 稠密向量列表
            sparse_vectors: 稀疏向量列表
            metadata: 元数据列表
            **kwargs: 其他参数
        """
        if not ids:
            return

        # 构建数据
        data = []
        for i in range(len(ids)):
            record = {
                "id": ids[i],
                "text": texts[i],
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
            }
            # 添加元数据（动态字段）
            if metadata and i < len(metadata):
                record.update(metadata[i])

            data.append(record)

        # 插入数据
        self.client.upsert(
            collection_name=self.collection_name,
            data=data
        )

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """稠密向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_expr: 过滤表达式
            **kwargs: 其他参数

        Returns:
            检索结果列表
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            limit=top_k,
            filter=filter_expr,
            output_fields=["id", "text"],
            **kwargs
        )

        # 格式化结果
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "id": hit["entity"]["id"],
                "text": hit["entity"]["text"],
                "score": hit["distance"],
                "metadata": {k: v for k, v in hit["entity"].items() if k not in ["id", "text"]}
            })

        return formatted_results

    def search_sparse(
        self,
        query_vector: Dict[str, Any],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """稀疏向量检索

        Args:
            query_vector: 查询向量（格式：{"indices": [...], "values": [...]}）
            top_k: 返回结果数量
            filter_expr: 过滤表达式
            **kwargs: 其他参数

        Returns:
            检索结果列表
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="sparse_vector",
            limit=top_k,
            filter=filter_expr,
            output_fields=["id", "text"],
            **kwargs
        )

        # 格式化结果
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "id": hit["entity"]["id"],
                "text": hit["entity"]["text"],
                "score": hit["distance"],
                "metadata": {k: v for k, v in hit["entity"].items() if k not in ["id", "text"]}
            })

        return formatted_results

    def get_by_ids(self, ids: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """根据 ID 获取记录

        Args:
            ids: 文档 ID 列表
            **kwargs: 其他参数

        Returns:
            记录列表
        """
        results = self.client.get(
            collection_name=self.collection_name,
            ids=ids,
            output_fields=["id", "text"]
        )

        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": {k: v for k, v in r.items() if k not in ["id", "text"]}
            }
            for r in results
        ]

    def delete(self, ids: List[str], **kwargs: Any) -> None:
        """删除记录

        Args:
            ids: 文档 ID 列表
            **kwargs: 其他参数
        """
        if not ids:
            return

        # 构建过滤表达式
        id_list_str = ", ".join([f'"{id_}"' for id_ in ids])
        filter_expr = f"id in [{id_list_str}]"

        self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expr
        )

    def count(self, **kwargs: Any) -> int:
        """获取记录总数

        Args:
            **kwargs: 其他参数

        Returns:
            记录数量
        """
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return stats["row_count"]
