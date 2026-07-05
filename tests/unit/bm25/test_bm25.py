#!/usr/bin/env python3
"""Test BM25 encoder with jieba tokenization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder

# 加载重建的 BM25 编码器
encoder = BM25SparseEncoder.load('data/db/bm25_encoder.pkl')

print(f'词表大小: {len(encoder.vocab)}')
print(f'词表前 20 个词: {list(encoder.vocab.keys())[:20]}')
print()

# 测试 "扫地机器人"
query = '扫地机器人'
vector = encoder.encode(query)
print(f'查询词: "{query}"')
print(f'Sparse vector: {vector}')
print(f'向量维度数: {len(vector)}')
print()

# 测试 "智能扫地机器人V3"
query2 = '智能扫地机器人V3'
vector2 = encoder.encode(query2)
print(f'查询词: "{query2}"')
print(f'Sparse vector: {vector2}')
print(f'向量维度数: {len(vector2)}')
