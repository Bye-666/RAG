"""MilvusStore 单元测试"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_pymilvus():
    """Mock pymilvus 模块"""
    with patch.dict("sys.modules", {"pymilvus": MagicMock()}):
        yield


class TestMilvusStore:
    """MilvusStore 测试"""

    def test_import(self):
        """测试导入模块"""
        from src.libs.vector_store.milvus_store import MilvusStore
        assert MilvusStore is not None

    def test_init_and_inheritance(self, mock_pymilvus):
        """测试初始化和继承关系"""
        from src.libs.vector_store.milvus_store import MilvusStore
        from src.libs.vector_store.base import BaseVectorStore

        # 验证继承关系
        assert issubclass(MilvusStore, BaseVectorStore)

    def test_has_required_methods(self, mock_pymilvus):
        """测试包含所有必需方法"""
        from src.libs.vector_store.milvus_store import MilvusStore

        required_methods = [
            "upsert", "search_dense", "search_sparse",
            "get_by_ids", "delete", "count"
        ]

        for method in required_methods:
            assert hasattr(MilvusStore, method)
            assert callable(getattr(MilvusStore, method))

    def test_default_parameters(self, mock_pymilvus):
        """测试默认参数"""
        from src.libs.vector_store.milvus_store import MilvusStore

        # 通过检查__init__签名验证默认参数
        import inspect
        sig = inspect.signature(MilvusStore.__init__)

        assert sig.parameters["uri"].default == "./data/db/milvus.db"
        assert sig.parameters["collection_name"].default == "rag_collection"
        assert sig.parameters["dense_dim"].default == 2048

    def test_repr_method_exists(self, mock_pymilvus):
        """测试字符串表示方法存在"""
        from src.libs.vector_store.milvus_store import MilvusStore

        # MilvusStore 继承自 BaseVectorStore，应该有 __repr__
        assert hasattr(MilvusStore, "__repr__")

    def test_search_sparse_lite_loads_collection_before_search(self):
        """Milvus Lite sparse search loads the collection before searching."""
        from src.libs.vector_store.milvus_store import MilvusStore

        store = MilvusStore.__new__(MilvusStore)
        store.collection_name = "test_collection"
        store._lite_server = MagicMock()

        collection = MagicMock()
        collection.search.return_value = [[
            {"id": "doc1", "distance": -2.0, "entity": {"text": "Vue text"}}
        ]]
        store._lite_server.get_collection.return_value = collection

        query_vector = {1: 1.0}
        results = store.search_sparse(query_vector=query_vector, top_k=3)

        collection.load.assert_called_once()
        collection.search.assert_called_once_with(
            query_vectors=[query_vector],
            top_k=3,
            metric_type="IP",
            expr=None,
            output_fields=["id", "text"],
            anns_field="sparse_vector"
        )
        assert results == [{
            "id": "doc1",
            "text": "Vue text",
            "score": -2.0,
            "metadata": {}
        }]

