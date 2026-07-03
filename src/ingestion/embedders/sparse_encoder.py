"""BM25 稀疏向量编码器

基于 BM25 算法生成稀疏向量，用于混合检索。
"""
import pickle
from collections import Counter
from typing import Dict, List

import numpy as np


class BM25SparseEncoder:
    """BM25 稀疏编码器

    在文档集合上训练 IDF，为每个文档生成稀疏向量。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """初始化编码器

        Args:
            k1: BM25 参数 k1
            b: BM25 参数 b
        """
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = {}
        self.avgdl = 0.0
        self.doc_count = 0

    def fit(self, texts: List[str]) -> None:
        """在文档集合上训练

        Args:
            texts: 文档文本列表
        """
        self.doc_count = len(texts)
        doc_lengths = []
        term_doc_freq = Counter()

        # 统计词频和文档长度
        for text in texts:
            terms = self._tokenize(text)
            doc_lengths.append(len(terms))
            unique_terms = set(terms)
            for term in unique_terms:
                term_doc_freq[term] += 1

        # 计算平均文档长度
        self.avgdl = np.mean(doc_lengths)

        # 构建词表和 IDF
        for idx, (term, df) in enumerate(term_doc_freq.items()):
            self.vocab[term] = idx
            # IDF 公式
            self.idf[idx] = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def encode(self, text: str) -> Dict[str, List]:
        """将文本编码为稀疏向量

        Args:
            text: 输入文本

        Returns:
            稀疏向量格式：{"indices": [...], "values": [...]}
        """
        terms = self._tokenize(text)
        term_freq = Counter(terms)
        doc_len = len(terms)

        indices = []
        values = []

        for term, tf in term_freq.items():
            if term in self.vocab:
                term_id = self.vocab[term]
                idf = self.idf[term_id]

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = idf * (numerator / denominator)

                indices.append(term_id)
                values.append(float(score))

        return {"indices": indices, "values": values}

    def _tokenize(self, text: str) -> List[str]:
        """分词

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        # 简单空格分词（生产环境应使用 jieba）
        return text.lower().split()

    def save(self, path: str) -> None:
        """持久化编码器

        Args:
            path: 保存路径
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'idf': self.idf,
                'avgdl': self.avgdl,
                'doc_count': self.doc_count,
                'k1': self.k1,
                'b': self.b
            }, f)

    @classmethod
    def load(cls, path: str) -> 'BM25SparseEncoder':
        """加载已训练的编码器

        Args:
            path: 模型路径

        Returns:
            编码器实例
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        encoder = cls(k1=data['k1'], b=data['b'])
        encoder.vocab = data['vocab']
        encoder.idf = data['idf']
        encoder.avgdl = data['avgdl']
        encoder.doc_count = data['doc_count']
        return encoder
