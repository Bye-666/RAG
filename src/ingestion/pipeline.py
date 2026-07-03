"""Ingestion Pipeline

数据摄取流水线，协调各组件完成文档到向量的转换。
"""
from typing import List, Optional

from .types import Document, Chunk, ChunkRecord
from .loaders.base import BaseLoader
from .embedders.dense_embedder import DenseEmbedder
from .embedders.sparse_encoder import BM25SparseEncoder
from ..libs.splitter.base import BaseSplitter
from ..libs.vector_store.base import BaseVectorStore


class IngestionPipeline:
    """数据摄取流水线

    协调 Loader -> Splitter -> Embedder -> VectorStore。
    """

    def __init__(
        self,
        loader: BaseLoader,
        splitter: BaseSplitter,
        dense_embedder: DenseEmbedder,
        sparse_encoder: BM25SparseEncoder,
        vector_store: BaseVectorStore
    ):
        """初始化 Pipeline

        Args:
            loader: 文档加载器
            splitter: 文本分块器
            dense_embedder: 稠密向量编码器
            sparse_encoder: 稀疏向量编码器
            vector_store: 向量存储
        """
        self.loader = loader
        self.splitter = splitter
        self.dense_embedder = dense_embedder
        self.sparse_encoder = sparse_encoder
        self.vector_store = vector_store

    def ingest_file(self, file_path: str) -> int:
        """摄取单个文件

        Args:
            file_path: 文件路径

        Returns:
            插入的 chunk 数量
        """
        # 1. 加载文档
        doc = self.loader.load(file_path)

        # 2. 分块
        chunk_texts = self.splitter.split(doc.text)

        # 3. 创建 Chunks
        chunks = []
        for idx, text in enumerate(chunk_texts):
            chunk = Chunk(
                id=f"{doc.id}_chunk_{idx}",
                text=text,
                doc_id=doc.id,
                chunk_index=idx,
                metadata={**doc.metadata, "chunk_index": idx}
            )
            chunks.append(chunk)

        # 4. 向量化
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]

        # 稠密向量
        dense_vectors = self.dense_embedder.encode_batch(texts)

        # 稀疏向量
        sparse_vectors = [self.sparse_encoder.encode(text) for text in texts]

        # 5. 存储
        metadata_list = [c.metadata for c in chunks]
        self.vector_store.upsert(
            ids=ids,
            texts=texts,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            metadata=metadata_list
        )

        return len(chunks)

    def ingest_batch(self, file_paths: List[str]) -> int:
        """批量摄取文件

        Args:
            file_paths: 文件路径列表

        Returns:
            总插入的 chunk 数量
        """
        total = 0
        for file_path in file_paths:
            count = self.ingest_file(file_path)
            total += count
        return total
