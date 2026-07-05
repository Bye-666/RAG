#!/usr/bin/env python3
"""Test BM25 partial_fit incremental update."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder

# 模拟场景：第一批文档
print("=== 初始训练 ===")
docs1 = [
    "智能扫地机器人V3 支持自动清洁",
    "智能音箱 支持语音控制"
]

encoder = BM25SparseEncoder()
encoder.fit(docs1)

print(f"词表大小: {len(encoder.vocab)}")
print(f"词表: {list(encoder.vocab.keys())}")
print(f"文档数: {encoder.doc_count}")
print()

# 测试查询 "扫地机器人"
query1 = "扫地机器人"
vector1 = encoder.encode(query1)
print(f'查询 "{query1}": {vector1} (维度数: {len(vector1)})')

# 测试查询 "电视" (新词，不在词表中)
query2 = "智能电视"
vector2 = encoder.encode(query2)
print(f'查询 "{query2}": {vector2} (维度数: {len(vector2)})')
print()

# 模拟场景：增量添加第二批文档
print("=== 增量更新 ===")
docs2 = [
    "智能电视 支持4K画质",
    "智能冰箱 可以语音购物"
]

encoder.partial_fit(docs2)

print(f"词表大小: {len(encoder.vocab)}")
print(f"新增词表: {list(encoder.vocab.keys())}")
print(f"文档数: {encoder.doc_count}")
print()

# 再次测试查询 "智能电视" (现在应该有结果了)
query3 = "智能电视"
vector3 = encoder.encode(query3)
print(f'查询 "{query3}": {vector3} (维度数: {len(vector3)})')

# 测试查询 "冰箱"
query4 = "智能冰箱"
vector4 = encoder.encode(query4)
print(f'查询 "{query4}": {vector4} (维度数: {len(vector4)})')
print()

print("=== 验证结果 ===")
print(f"✓ 增量更新前 '智能电视' 维度: {len(vector2)}")
print(f"✓ 增量更新后 '智能电视' 维度: {len(vector3)}")
print(f"✓ 新词 '智能冰箱' 维度: {len(vector4)}")

if len(vector3) > len(vector2):
    print("\n✓ 增量更新成功！新词已加入词表并可被检索")
else:
    print("\n✗ 增量更新失败")
