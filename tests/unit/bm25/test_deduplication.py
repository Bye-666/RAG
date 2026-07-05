#!/usr/bin/env python3
"""Test BM25 encoder document deduplication."""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path.cwd()))

from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder

print("=== Test 1: Deduplication in fit() ===")

docs_with_dup = [
    "智能 扫地 机器人",
    "智能 音箱",
    "智能 扫地 机器人",  # duplicate
    "智能 电视"
]

encoder = BM25SparseEncoder()
encoder.fit(docs_with_dup)

print(f"Input: {len(docs_with_dup)} documents (1 duplicate)")
print(f"Result: doc_count={encoder.doc_count}, vocab_size={len(encoder.vocab)}")
print(f"doc_hashes: {len(encoder.doc_hashes)} unique hashes")

assert encoder.doc_count == 3, f"Expected 3 unique docs, got {encoder.doc_count}"
print("[OK] fit() correctly deduplicates documents")

print("\n=== Test 2: Deduplication in partial_fit() ===")

new_docs_with_dup = [
    "智能 冰箱",
    "智能 电视",  # duplicate from fit()
    "智能 冰箱",  # duplicate in same batch
    "智能 洗衣机"
]

new_count = encoder.partial_fit(new_docs_with_dup)

print(f"Input: {len(new_docs_with_dup)} documents (2 duplicates)")
print(f"Result: {new_count} new documents added")
print(f"Total doc_count: {encoder.doc_count}")
print(f"Total unique hashes: {len(encoder.doc_hashes)}")

assert new_count == 2, f"Expected 2 new docs, got {new_count}"
assert encoder.doc_count == 5, f"Expected 5 total docs, got {encoder.doc_count}"
print("[OK] partial_fit() correctly deduplicates and returns new doc count")

print("\n=== Test 3: Deduplication persists across save/load ===")

# Save
with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
    temp_path = tmp.name

encoder.save(temp_path)
print(f"Saved encoder with {len(encoder.doc_hashes)} doc hashes")

# Load
encoder2 = BM25SparseEncoder.load(temp_path)
print(f"Loaded encoder with {len(encoder2.doc_hashes)} doc hashes")

# Try to add duplicates again
dup_docs = [
    "智能 扫地 机器人",  # from original fit()
    "智能 冰箱",  # from partial_fit()
    "智能 空调"  # new
]

new_count2 = encoder2.partial_fit(dup_docs)

print(f"Input: {len(dup_docs)} documents (2 duplicates)")
print(f"Result: {new_count2} new documents added")
print(f"Total doc_count: {encoder2.doc_count}")

assert new_count2 == 1, f"Expected 1 new doc, got {new_count2}"
assert encoder2.doc_count == 6, f"Expected 6 total docs, got {encoder2.doc_count}"
print("[OK] doc_hashes persists across save/load")

print("\n=== Test 4: IDF calculation with deduplication ===")

encoder3 = BM25SparseEncoder()
encoder3.fit([
    "alpha beta",
    "alpha gamma",
    "alpha beta"  # duplicate - should not count twice
])

print(f"doc_count: {encoder3.doc_count}")
print(f"term_doc_freq: {encoder3.term_doc_freq}")

# 'alpha' appears in both unique docs: df=2
# 'beta' appears in 1 unique doc: df=1
# 'gamma' appears in 1 unique doc: df=1
assert encoder3.term_doc_freq.get('alpha') == 2, "alpha should appear in 2 docs"
assert encoder3.term_doc_freq.get('beta') == 1, "beta should appear in 1 doc"
assert encoder3.doc_count == 2, "Should have 2 unique docs"
print("[OK] IDF correctly calculated with deduplication")

# Cleanup
Path(temp_path).unlink()

print("\n=== All Deduplication Tests Passed ===")
