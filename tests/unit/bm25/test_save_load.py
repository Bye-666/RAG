#!/usr/bin/env python3
"""Test BM25 encoder save/load persistence of incremental stats."""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path.cwd()))

from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder

print("=== Test 1: Save/Load Persistence ===")

# Initial training
docs1 = ["智能 扫地 机器人", "智能 音箱"]
encoder1 = BM25SparseEncoder()
encoder1.fit(docs1)

print(f"Initial state:")
print(f"  vocab size: {len(encoder1.vocab)}")
print(f"  doc_count: {encoder1.doc_count}")
print(f"  term_doc_freq: {encoder1.term_doc_freq}")
print(f"  total_doc_length: {encoder1.total_doc_length}")
print(f"  avgdl: {encoder1.avgdl}")

# Save to temp file
with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
    temp_path = tmp.name

encoder1.save(temp_path)
print(f"\nSaved to {temp_path}")

# Load and verify
encoder2 = BM25SparseEncoder.load(temp_path)
print(f"\nLoaded state:")
print(f"  vocab size: {len(encoder2.vocab)}")
print(f"  doc_count: {encoder2.doc_count}")
print(f"  term_doc_freq: {encoder2.term_doc_freq}")
print(f"  total_doc_length: {encoder2.total_doc_length}")
print(f"  avgdl: {encoder2.avgdl}")

# Verify persistence
assert encoder2.vocab == encoder1.vocab, "vocab mismatch"
assert encoder2.term_doc_freq == encoder1.term_doc_freq, "term_doc_freq mismatch"
assert encoder2.total_doc_length == encoder1.total_doc_length, "total_doc_length mismatch"
assert encoder2.doc_count == encoder1.doc_count, "doc_count mismatch"
print("\n[OK] Save/load preserves all incremental stats")

print("\n=== Test 2: Partial Fit After Load ===")

# Incremental update after load
docs2 = ["智能 电视", "智能 冰箱"]
encoder2.partial_fit(docs2)

print(f"\nAfter partial_fit:")
print(f"  vocab size: {len(encoder2.vocab)}")
print(f"  doc_count: {encoder2.doc_count}")
print(f"  term_doc_freq: {encoder2.term_doc_freq}")
print(f"  total_doc_length: {encoder2.total_doc_length}")
print(f"  avgdl: {encoder2.avgdl}")

# Verify old terms still have correct df
assert encoder2.term_doc_freq.get('智能', 0) == 4, "Old term '智能' should appear in 4 docs"
assert encoder2.term_doc_freq.get('扫地', 0) == 1, "Old term '扫地' should appear in 1 doc"
assert encoder2.term_doc_freq.get('电视', 0) == 1, "New term '电视' should appear in 1 doc"
print("\n[OK] Partial fit after load works correctly")

print("\n=== Test 3: Fit Resets State ===")

encoder3 = BM25SparseEncoder()
encoder3.fit(["alpha beta"])
print(f"\nAfter first fit: vocab={list(encoder3.vocab.keys())}")

encoder3.fit(["gamma delta"])
print(f"After second fit: vocab={list(encoder3.vocab.keys())}")

assert 'alpha' not in encoder3.vocab, "fit() should reset and remove old terms"
assert 'gamma' in encoder3.vocab, "fit() should contain new terms"
print("\n[OK] fit() correctly resets all state")

# Cleanup
Path(temp_path).unlink()
print("\n=== All Tests Passed ===")
