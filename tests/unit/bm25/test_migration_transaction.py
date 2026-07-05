#!/usr/bin/env python3
"""Test old pickle migration and transactional BM25 updates."""
import sys
from pathlib import Path
import tempfile
import pickle

sys.path.insert(0, str(Path.cwd()))

from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder

print("=== Test 1: Old Pickle Format Migration ===")

# Create an old format pickle (without incremental stats)
old_encoder = BM25SparseEncoder()
old_encoder.fit(["alpha beta", "alpha gamma"])

with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
    old_pickle_path = tmp.name

# Save in old format (manually construct old pickle)
old_data = {
    'vocab': old_encoder.vocab,
    'idf': old_encoder.idf,
    'avgdl': old_encoder.avgdl,
    'doc_count': old_encoder.doc_count,
    'k1': old_encoder.k1,
    'b': old_encoder.b
    # Missing: term_doc_freq, total_doc_length, doc_hashes
}

with open(old_pickle_path, 'wb') as f:
    pickle.dump(old_data, f)

print(f"Created old format pickle: {old_pickle_path}")
print(f"  vocab: {old_data['vocab']}")
print(f"  doc_count: {old_data['doc_count']}")

# Load old pickle
print("\nLoading old format pickle...")
loaded_encoder = BM25SparseEncoder.load(old_pickle_path)

print(f"Loaded encoder state:")
print(f"  vocab: {loaded_encoder.vocab}")
print(f"  doc_count: {loaded_encoder.doc_count}")
print(f"  term_doc_freq: {loaded_encoder.term_doc_freq}")
print(f"  doc_hashes: {loaded_encoder.doc_hashes}")
print(f"  _incremental_stats_available: {loaded_encoder._incremental_stats_available}")

assert not loaded_encoder._incremental_stats_available, "Should detect old format"
print("\n[OK] Old format correctly detected")

# Try partial_fit on old encoder - should raise RuntimeError
print("\nAttempting partial_fit on old format encoder...")
try:
    loaded_encoder.partial_fit(["delta epsilon"])
    print("[FAIL] Should have raised RuntimeError")
    sys.exit(1)
except RuntimeError as e:
    print(f"[OK] Correctly rejected: {str(e)[:80]}...")

print("\n=== Test 2: New Pickle Format ===")

# Create new format pickle
new_encoder = BM25SparseEncoder()
new_encoder.fit(["alpha beta", "alpha gamma"])

with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
    new_pickle_path = tmp.name

new_encoder.save(new_pickle_path)

# Load new format
loaded_new = BM25SparseEncoder.load(new_pickle_path)

print(f"Loaded new format encoder:")
print(f"  vocab: {loaded_new.vocab}")
print(f"  term_doc_freq: {loaded_new.term_doc_freq}")
print(f"  doc_hashes: {len(loaded_new.doc_hashes)} hashes")
print(f"  _incremental_stats_available: {loaded_new._incremental_stats_available}")

assert loaded_new._incremental_stats_available, "Should detect new format"
print("\n[OK] New format correctly detected")

# partial_fit should work
new_count = loaded_new.partial_fit(["delta epsilon"])
print(f"\n[OK] partial_fit succeeded: {new_count} new docs added")

print("\n=== Test 3: Transactional Update Simulation ===")

# Simulate: load encoder, prepare update, but DON'T save on failure
encoder = BM25SparseEncoder()
encoder.fit(["doc1 content"])

with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
    test_path = tmp.name

encoder.save(test_path)
original_doc_count = encoder.doc_count
print(f"Initial encoder: doc_count={original_doc_count}")

# Simulate ingestion failure scenario
print("\nSimulating ingestion failure...")
encoder_copy = BM25SparseEncoder.load(test_path)
print(f"Loaded encoder: doc_count={encoder_copy.doc_count}")

# Prepare update (but ingestion fails, so we DON'T save)
encoder_copy.partial_fit(["doc2 content"])
print(f"After partial_fit: doc_count={encoder_copy.doc_count}")
print("*** Simulating ingestion failure - NOT saving encoder ***")

# Reload from disk - should still have original state
encoder_reloaded = BM25SparseEncoder.load(test_path)
print(f"\nReloaded encoder: doc_count={encoder_reloaded.doc_count}")

assert encoder_reloaded.doc_count == original_doc_count, "Should preserve original state"
print("[OK] Failed ingestion did not corrupt saved encoder")

# Cleanup
Path(old_pickle_path).unlink()
Path(new_pickle_path).unlink()
Path(test_path).unlink()

print("\n=== All Migration and Transaction Tests Passed ===")
