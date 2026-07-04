"""DashScope (阿里云千问) Embedding 实现

支持的模型：
- text-embedding-v4: 2048维（默认，推荐）
- text-embedding-v2: 1536维
- text-embedding-v1: 1536维
"""
import os
from typing import Any, List, Optional

from .base import BaseEmbedding


class DashScopeEmbedding(BaseEmbedding):
    """DashScope Embedding 实现

    使用阿里云 DashScope API 调用千问系列 Embedding 模型。
    """

    # 模型维度映射
    MODEL_DIMENSIONS = {
        "text-embedding-v4": 2048,
        "text-embedding-v2": 1536,
        "text-embedding-v1": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-v4",
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """初始化 DashScope Embedding

        Args:
            model: 模型名称，默认 text-embedding-v4
            api_key: DashScope API Key，如果为 None 则从环境变量获取
            **kwargs: 其他参数
        """
        super().__init__(model=model, api_key=api_key, **kwargs)

        # 获取 API Key
        if self.api_key is None:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DashScope API Key 未设置。请通过参数传入或设置环境变量 DASHSCOPE_API_KEY"
            )

        # 清除代理设置（避免代理连接问题）
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)

        # 延迟导入
        try:
            import dashscope
            self.dashscope = dashscope
            self.dashscope.api_key = self.api_key
        except ImportError:
            raise ImportError(
                "dashscope 包未安装。请运行：pip install dashscope"
            )

    def encode(self, text: str, **kwargs: Any) -> List[float]:
        """编码单个文本为向量

        Args:
            text: 输入文本
            **kwargs: 其他编码参数

        Returns:
            向量（浮点数列表）

        Raises:
            Exception: API 调用失败时抛出异常
        """
        from dashscope import TextEmbedding

        # 合并参数
        embedding_kwargs = {**self.kwargs, **kwargs}

        # 调用 API
        response = TextEmbedding.call(
            model=self.model,
            input=text,
            **embedding_kwargs
        )

        # 检查响应
        if response.status_code != 200:
            raise Exception(
                f"DashScope API 调用失败: {response.code} - {response.message}"
            )

        # 提取向量（response.output 是字典）
        embeddings = response.output.get('embeddings', [])
        if not embeddings:
            raise Exception("DashScope API 返回空的 embeddings")

        # 返回第一个 embedding（可能是对象或字典）
        first_embedding = embeddings[0]
        if isinstance(first_embedding, dict):
            return first_embedding.get('embedding', [])
        else:
            return first_embedding.embedding

    def embed(self, text: str, **kwargs: Any) -> List[float]:
        """编码单个文本为向量（encode 的别名）

        Args:
            text: 输入文本
            **kwargs: 其他编码参数

        Returns:
            向量（浮点数列表）
        """
        return self.encode(text, **kwargs)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
        **kwargs: Any
    ) -> List[List[float]]:
        """批量编码文本为向量

        DashScope API 限制每次最多 10 个文本，自动分批处理。

        Args:
            texts: 输入文本列表
            batch_size: 每批处理的文本数量，默认 10（DashScope 限制）
            **kwargs: 其他编码参数

        Returns:
            向量列表

        Raises:
            Exception: API 调用失败时抛出异常
        """
        from dashscope import TextEmbedding

        # 合并参数
        embedding_kwargs = {**self.kwargs, **kwargs}

        # 限制批量大小不超过 10
        max_batch_size = min(batch_size, 10)

        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]

            # 调用 API（批量）
            response = TextEmbedding.call(
                model=self.model,
                input=batch,
                **embedding_kwargs
            )

            # 检查响应
            if response.status_code != 200:
                raise Exception(
                    f"DashScope API 调用失败: {response.code} - {response.message}"
                )

            # 提取向量列表（response.output 是字典）
            embeddings = response.output.get('embeddings', [])
            if not embeddings:
                raise Exception("DashScope API 返回空的 embeddings")

            # 添加到结果（可能是对象或字典）
            for emb in embeddings:
                if isinstance(emb, dict):
                    all_embeddings.append(emb.get('embedding', []))
                else:
                    all_embeddings.append(emb.embedding)

        return all_embeddings

    @property
    def dimension(self) -> int:
        """向量维度

        Returns:
            向量维度
        """
        return self.MODEL_DIMENSIONS.get(self.model, 2048)
