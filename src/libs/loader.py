"""配置驱动的组件加载器

从配置文件加载并实例化所有组件。
"""
from typing import Any, Dict, Optional

from .factory import ComponentFactory


class ComponentLoader:
    """组件加载器

    从配置文件读取配置，并使用工厂创建组件实例。
    """

    def __init__(self, settings):
        """初始化加载器

        Args:
            settings: Settings 实例
        """
        self.settings = settings
        self._llm = None
        self._embedding = None
        self._vision_llm = None
        self._vector_store = None
        self._splitter = None
        self._reranker = None

    def get_llm(self):
        """获取 LLM 实例（懒加载）"""
        if self._llm is None:
            config = self.settings.get("llm", {})
            self._llm = ComponentFactory.create_llm(config)
        return self._llm

    def get_embedding(self):
        """获取 Embedding 实例（懒加载）"""
        if self._embedding is None:
            config = self.settings.get("embedding", {})
            self._embedding = ComponentFactory.create_embedding(config)
        return self._embedding

    def get_vision_llm(self):
        """获取 Vision LLM 实例（懒加载）"""
        if self._vision_llm is None:
            config = self.settings.get("vision_llm", {})
            self._vision_llm = ComponentFactory.create_vision_llm(config)
        return self._vision_llm

    def get_vector_store(self):
        """获取 Vector Store 实例（懒加载）"""
        if self._vector_store is None:
            config = self.settings.get("vector_store", {})
            self._vector_store = ComponentFactory.create_vector_store(config)
        return self._vector_store

    def get_splitter(self):
        """获取 Splitter 实例（懒加载）"""
        if self._splitter is None:
            config = self.settings.get("splitter", {})
            self._splitter = ComponentFactory.create_splitter(config)
        return self._splitter

    def get_reranker(self):
        """获取 Reranker 实例（懒加载）"""
        if self._reranker is None:
            config = self.settings.get("rerank", {})
            self._reranker = ComponentFactory.create_reranker(config)
        return self._reranker
