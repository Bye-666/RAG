"""工厂模式

根据配置动态创建组件实例。
"""
from typing import Any, Dict, Optional


class ComponentFactory:
    """组件工厂

    根据配置文件动态创建各种组件实例。
    """

    @staticmethod
    def create_llm(config: Dict[str, Any]) -> Any:
        """创建 LLM 实例

        Args:
            config: LLM 配置

        Returns:
            LLM 实例
        """
        provider = config.get("provider", "dashscope")

        if provider == "dashscope":
            from .llm.dashscope_llm import DashScopeLLM
            return DashScopeLLM(
                model=config.get("model", "qwen-max"),
                api_key=config.get("api_key")
            )
        else:
            raise ValueError(f"不支持的 LLM provider: {provider}")

    @staticmethod
    def create_embedding(config: Dict[str, Any]) -> Any:
        """创建 Embedding 实例

        Args:
            config: Embedding 配置

        Returns:
            Embedding 实例
        """
        provider = config.get("provider", "dashscope")

        if provider == "dashscope":
            from .embedding.dashscope_embedding import DashScopeEmbedding
            return DashScopeEmbedding(
                model=config.get("model", "text-embedding-v4"),
                api_key=config.get("api_key")
            )
        else:
            raise ValueError(f"不支持的 Embedding provider: {provider}")

    @staticmethod
    def create_vision_llm(config: Dict[str, Any]) -> Any:
        """创建 Vision LLM 实例

        Args:
            config: Vision LLM 配置

        Returns:
            Vision LLM 实例
        """
        provider = config.get("provider", "dashscope")

        if provider == "dashscope":
            from .vision_llm.dashscope_vision import DashScopeVision
            return DashScopeVision(
                model=config.get("model", "qwen-vl-max"),
                api_key=config.get("api_key")
            )
        else:
            raise ValueError(f"不支持的 Vision LLM provider: {provider}")

    @staticmethod
    def create_vector_store(config: Dict[str, Any]) -> Any:
        """创建 Vector Store 实例

        Args:
            config: Vector Store 配置

        Returns:
            Vector Store 实例
        """
        store_type = config.get("type", "milvus")

        if store_type == "milvus":
            from .vector_store.milvus_store import MilvusStore
            return MilvusStore(
                uri=config.get("uri", "./data/db/milvus.db"),
                collection_name=config.get("collection_name", "rag_collection"),
                dense_dim=config.get("dense_dim", 2048)
            )
        else:
            raise ValueError(f"不支持的 Vector Store type: {store_type}")

    @staticmethod
    def create_splitter(config: Dict[str, Any]) -> Any:
        """创建 Splitter 实例

        Args:
            config: Splitter 配置

        Returns:
            Splitter 实例
        """
        from .splitter.recursive_splitter import RecursiveSplitter

        return RecursiveSplitter(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200)
        )

    @staticmethod
    def create_reranker(config: Dict[str, Any]) -> Optional[Any]:
        """创建 Reranker 实例

        Args:
            config: Reranker 配置

        Returns:
            Reranker 实例，如果未配置则返回 None
        """
        backend = config.get("backend", "none")

        if backend == "none":
            return None
        elif backend == "cross_encoder":
            from .reranker.cross_encoder import CrossEncoderReranker
            return CrossEncoderReranker(
                model=config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )
        else:
            raise ValueError(f"不支持的 Reranker backend: {backend}")
