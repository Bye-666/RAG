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
        self._lite_server = None

        # 延迟导入
        try:
            from pymilvus import MilvusClient, DataType, connections
            self.MilvusClient = MilvusClient
            self.DataType = DataType
            self.connections = connections
        except ImportError:
            raise ImportError(
                "pymilvus 包未安装。请运行：pip install pymilvus"
            )

        # 检查是否是本地文件模式
        if self._is_local_file(uri):
            # 启动 Milvus Lite 本地服务器
            self._start_lite_server()
            # 连接到本地服务器
            self.client = None  # 使用 connections API 而不是 MilvusClient
            self._connect_to_lite()
        else:
            # 连接到远程 Milvus 服务
            self.client = MilvusClient(uri=self.uri)

        # 确保集合存在
        self._ensure_collection()

    def _is_local_file(self, uri: str) -> bool:
        """检查 URI 是否是本地文件路径"""
        return not uri.startswith("http://") and not uri.startswith("https://")

    def _start_lite_server(self) -> None:
        """启动 Milvus Lite 本地服务器"""
        try:
            from milvus_lite import MilvusLite
            import os
            from pathlib import Path

            # 转换为绝对路径（Windows 上避免路径问题）
            db_path = Path(self.uri).resolve()

            # 确保目录存在
            db_dir = db_path.parent
            if not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)

            # 启动 Milvus Lite（使用绝对路径）
            self._lite_server = MilvusLite(str(db_path))
            self.uri = str(db_path)  # 更新为绝对路径
        except ImportError:
            raise ImportError(
                "milvus-lite 包未安装。请运行：pip install milvus-lite"
            )

    def _connect_to_lite(self) -> None:
        """连接到 Milvus Lite 服务器"""
        # 使用 connections API 建立默认连接
        # Milvus Lite 内嵌在进程中，不需要 URI
        # 但需要创建一个连接别名供 Collection API 使用
        try:
            # 尝试连接到已启动的 Milvus Lite
            # 由于 Milvus Lite 是内嵌的，我们需要使用特殊的方式
            from pymilvus import connections

            # 不进行显式连接，Milvus Lite 通过直接 API 调用工作
            # 但 Collection/utility 需要一个默认连接
            # 这里我们跳过连接，直接使用 Collection
        except Exception as e:
            print(f"Warning: Could not establish connection: {e}")

    def _ensure_collection(self) -> None:
        """确保集合存在，如果不存在则创建"""
        if self._lite_server is not None:
            # Milvus Lite 模式
            self._ensure_collection_lite()
        else:
            # 标准 Milvus 模式
            self._ensure_collection_standard()

    def _ensure_collection_standard(self) -> None:
        """标准 Milvus 模式创建集合"""
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

    def _ensure_collection_lite(self) -> None:
        """Milvus Lite 模式创建集合"""
        # Milvus Lite 使用直接 API，不需要 pymilvus 的 Collection
        # 直接操作 _lite_server 对象
        try:
            # 检查集合是否存在
            collections = self._lite_server.list_collections()
            if self.collection_name in collections:
                return

            # 使用 Milvus Lite 的 API 创建集合
            from milvus_lite import FieldSchema, CollectionSchema, DataType
            import time

            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            ]

            # 创建 Schema
            schema = CollectionSchema(
                fields=fields,
                enable_dynamic_field=True
            )

            # 创建集合
            collection = self._lite_server.create_collection(
                name=self.collection_name,
                schema=schema
            )

            # 创建索引时添加重试逻辑（Windows 文件系统问题）
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    collection.create_index(
                        field_name="dense_vector",
                        index_params={
                            "index_type": "FLAT",
                            "metric_type": "COSINE"
                        }
                    )
                    break
                except Exception as e:
                    # 如果索引已存在，跳过
                    if "already exists" in str(e):
                        break
                    if attempt < max_retries - 1:
                        print(f"[WARNING] Index creation retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(0.5)  # 等待文件系统同步
                    else:
                        print(f"[WARNING] Dense index creation failed after {max_retries} attempts, continuing...")

            for attempt in range(max_retries):
                try:
                    collection.create_index(
                        field_name="sparse_vector",
                        index_params={
                            "index_type": "SPARSE_INVERTED_INDEX",
                            "metric_type": "IP"
                        }
                    )
                    break
                except Exception as e:
                    # 如果索引已存在，跳过
                    if "already exists" in str(e):
                        break
                    if attempt < max_retries - 1:
                        print(f"[WARNING] Index creation retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(0.5)
                    else:
                        print(f"[WARNING] Sparse index creation failed after {max_retries} attempts, continuing...")

        except Exception as e:
            print(f"Error creating collection in Milvus Lite: {e}")
            raise

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
        if self._lite_server is not None:
            # Milvus Lite 模式
            try:
                collection = self._lite_server.get_collection(self.collection_name)
                collection.insert(data)
            except Exception as e:
                print(f"Upsert error in Milvus Lite: {e}")
                raise
        else:
            # 标准 Milvus 模式
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
        if self._lite_server is not None:
            # Milvus Lite 模式 - 使用不同的 API
            try:
                collection = self._lite_server.get_collection(self.collection_name)

                # 确保 Collection 已加载
                try:
                    collection.load()
                except Exception:
                    pass  # 已经加载，忽略错误

                # Milvus Lite 使用不同的参数
                results = collection.search(
                    query_vectors=[query_vector],
                    top_k=top_k,
                    metric_type="COSINE",
                    expr=filter_expr,
                    output_fields=["id", "text"],
                    anns_field="dense_vector"
                )

                # 格式化结果
                formatted_results = []
                for hit in results[0]:
                    # Milvus Lite 返回 dict，文本可能在 entity 中
                    text = hit.get("text")
                    if text is None and "entity" in hit:
                        text = hit["entity"].get("text")

                    formatted_results.append({
                        "id": hit.get("id"),
                        "text": text,
                        "score": hit.get("distance"),
                        "metadata": {k: v for k, v in hit.items() if k not in ["id", "text", "distance", "entity"]}
                    })

                return formatted_results
            except Exception as e:
                print(f"Search dense error in Milvus Lite: {e}")
                import traceback
                traceback.print_exc()
                return []
        else:
            # 标准 Milvus 模式
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
        if self._lite_server is not None:
            # Milvus Lite 模式 - 使用不同的 API
            try:
                collection = self._lite_server.get_collection(self.collection_name)

                # Keep sparse search consistent with dense search: Milvus Lite
                # collections can be in "released" state after startup/index
                # operations, and search requires an explicit load().
                try:
                    collection.load()
                except Exception:
                    pass  # Already loaded or load is a no-op in this runtime.

                # Milvus Lite 稀疏向量搜索
                results = collection.search(
                    query_vectors=[query_vector],
                    top_k=top_k,
                    metric_type="IP",
                    expr=filter_expr,
                    output_fields=["id", "text"],
                    anns_field="sparse_vector"
                )

                # 格式化结果
                formatted_results = []
                for hit in results[0]:
                    # Milvus Lite 返回 dict，文本可能在 entity 中
                    text = hit.get("text")
                    if text is None and "entity" in hit:
                        text = hit["entity"].get("text")

                    formatted_results.append({
                        "id": hit.get("id"),
                        "text": text,
                        "score": hit.get("distance"),
                        "metadata": {k: v for k, v in hit.items() if k not in ["id", "text", "distance", "entity"]}
                    })

                return formatted_results
            except Exception as e:
                print(f"Search sparse error in Milvus Lite: {e}")
                import traceback
                traceback.print_exc()
                return []
        else:
            # 标准 Milvus 模式
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
        if self._lite_server is not None:
            # Milvus Lite 模式
            try:
                collection = self._lite_server.get_collection(self.collection_name)
                # 构建查询表达式
                id_list_str = ", ".join([f'"{id_}"' for id_ in ids])
                expr = f"id in [{id_list_str}]"

                results = collection.query(
                    expr=expr,
                    output_fields=["id", "text"]
                )

                return [
                    {
                        "id": r.get("id", ""),
                        "text": r.get("text", ""),
                        "metadata": {k: v for k, v in r.items() if k not in ["id", "text"]}
                    }
                    for r in results
                ]
            except Exception as e:
                print(f"Get by IDs error in Milvus Lite: {e}")
                return []
        else:
            # 标准 Milvus 模式
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

        if self._lite_server is not None:
            # Milvus Lite 模式
            try:
                collection = self._lite_server.get_collection(self.collection_name)
                collection.delete(expr=filter_expr)
            except Exception as e:
                print(f"Delete error in Milvus Lite: {e}")
                raise
        else:
            # 标准 Milvus 模式
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
        stats = self.get_collection_stats()
        return stats["row_count"]

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息

        Returns:
            统计信息字典
        """
        if self._lite_server is not None:
            # Milvus Lite 模式
            try:
                collection = self._lite_server.get_collection(self.collection_name)
                return {
                    "row_count": collection.num_entities,
                    "collection_name": self.collection_name
                }
            except Exception:
                return {
                    "row_count": 0,
                    "collection_name": self.collection_name
                }
        else:
            # 标准 Milvus 模式
            return self.client.get_collection_stats(collection_name=self.collection_name)

    def query(
        self,
        expr: str = "",
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """查询记录

        Args:
            expr: 过滤表达式，空字符串表示查询所有
            output_fields: 输出字段列表
            limit: 返回结果数量
            **kwargs: 其他参数

        Returns:
            查询结果列表
        """
        if self._lite_server is not None:
            # Milvus Lite 模式
            try:
                collection = self._lite_server.get_collection(self.collection_name)

                # 确保 Collection 已加载（如果已加载会自动跳过）
                try:
                    collection.load()
                except Exception:
                    pass  # 已经加载，忽略错误

                # 如果没有过滤条件，使用简单查询
                if not expr:
                    expr = "id != ''"  # 查询所有记录

                # Milvus Lite: 不传 output_fields 会返回所有字段（包括动态字段）
                # 如果用户指定了 output_fields，则使用用户指定的
                query_params = {
                    "expr": expr,
                    "limit": limit
                }
                if output_fields is not None:
                    query_params["output_fields"] = output_fields

                results = collection.query(**query_params)

                return [
                    {
                        "id": r.get("id", ""),
                        "text": r.get("text", ""),
                        "metadata": {k: v for k, v in r.items() if k not in ["id", "text", "dense_vector", "sparse_vector"]}
                    }
                    for r in results
                ]
            except Exception as e:
                print(f"Query error in Milvus Lite: {e}")
                import traceback
                traceback.print_exc()
                raise  # 重新抛出异常而不是返回空列表
        else:
            # 标准 Milvus 模式
            results = self.client.query(
                collection_name=self.collection_name,
                filter=expr if expr else None,
                output_fields=output_fields,
                limit=limit,
                **kwargs
            )

            return [
                {
                    "id": r.get("id", ""),
                    "text": r.get("text", ""),
                    "metadata": {k: v for k, v in r.items() if k not in ["id", "text"]}
                }
                for r in results
            ]
