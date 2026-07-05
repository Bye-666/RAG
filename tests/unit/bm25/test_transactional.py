#!/usr/bin/env python3
"""Test transactional BM25 update with clone() strategy."""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path.cwd()))

from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder

print("=== Test 1: Clone Creates Independent Copy ===")

# Create base encoder
base = BM25SparseEncoder()
base.fit(["alpha beta", "alpha gamma"])

print(f"Base encoder: vocab={list(base.vocab.keys())}, doc_count={base.doc_count}")

# Clone it
cloned = base.clone()

print(f"Cloned encoder: vocab={list(cloned.vocab.keys())}, doc_count={cloned.doc_count}")

# Modify clone
cloned.partial_fit(["delta epsilon"])

print(f"\nAfter modifying clone:")
print(f"  Base: doc_count={base.doc_count}, vocab_size={len(base.vocab)}")
print(f"  Clone: doc_count={cloned.doc_count}, vocab_size={len(cloned.vocab)}")

assert base.doc_count == 2, "Base should remain unchanged"
assert cloned.doc_count == 3, "Clone should be updated"
assert 'delta' not in base.vocab, "Base vocab should not have new terms"
assert 'delta' in cloned.vocab, "Clone vocab should have new terms"

print("[OK] Clone is independent from base")

print("\n=== Test 2: Simulated Transactional Ingestion ===")

# Setup: base encoder with some data
base_encoder = BM25SparseEncoder()
base_encoder.fit(["document one content"])

with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
    encoder_path = tmp.name

base_encoder.save(encoder_path)
print(f"Base encoder saved: doc_count={base_encoder.doc_count}")

# Simulate ingestion attempt 1: SUCCESS
print("\n--- Ingestion Attempt 1: SUCCESS ---")
loaded = BM25SparseEncoder.load(encoder_path)
temp_encoder = loaded.clone()

# Prepare encoder with new documents
new_chunks = ["document two content", "document three content"]
new_count = temp_encoder.partial_fit(new_chunks)
print(f"Temporary encoder prepared: {new_count} new docs, total={temp_encoder.doc_count}")

# Simulate successful ingestion
print("*** Simulating successful ingestion ***")
ingestion_success = True

if ingestion_success:
    # Commit: save temporary encoder
    temp_encoder.save(encoder_path)
    print(f"[COMMIT] Encoder saved with doc_count={temp_encoder.doc_count}")

# Verify committed state
reloaded = BM25SparseEncoder.load(encoder_path)
assert reloaded.doc_count == 3, f"Expected doc_count=3, got {reloaded.doc_count}"
print(f"[OK] Committed state persisted: doc_count={reloaded.doc_count}")

# Simulate ingestion attempt 2: FAILURE
print("\n--- Ingestion Attempt 2: FAILURE ---")
loaded2 = BM25SparseEncoder.load(encoder_path)
temp_encoder2 = loaded2.clone()

# Prepare encoder
new_chunks2 = ["document four content"]
new_count2 = temp_encoder2.partial_fit(new_chunks2)
print(f"Temporary encoder prepared: {new_count2} new docs, total={temp_encoder2.doc_count}")

# Simulate failed ingestion
print("*** Simulating failed ingestion ***")
ingestion_success2 = False

if ingestion_success2:
    temp_encoder2.save(encoder_path)
    print("[COMMIT] Encoder saved")
else:
    print("[ROLLBACK] Encoder NOT saved, temporary changes discarded")

# Verify original state preserved
reloaded2 = BM25SparseEncoder.load(encoder_path)
assert reloaded2.doc_count == 3, f"Expected doc_count=3 (unchanged), got {reloaded2.doc_count}"
print(f"[OK] Failed ingestion preserved original state: doc_count={reloaded2.doc_count}")

print("\n=== Test 3: Encoder Used During Ingestion Has New Vocab ===")

# This test verifies the key fix: encoder used by BatchProcessor has updated vocab
base3 = BM25SparseEncoder()
base3.fit(["existing term"])

print(f"Base encoder vocab: {list(base3.vocab.keys())}")

# Clone and update (simulating pre-ingestion preparation)
temp3 = base3.clone()
temp3.partial_fit(["new term"])

print(f"Temporary encoder vocab: {list(temp3.vocab.keys())}")

# Verify new term is in temporary encoder
assert 'new' in temp3.vocab, "New term should be in temporary encoder"
assert 'new' not in base3.vocab, "New term should NOT be in base encoder yet"

# Simulate encoding during ingestion
query_text = "new term"
sparse_vector = temp3.encode(query_text)

print(f"Encoding '{query_text}' with temporary encoder: {sparse_vector}")
assert len(sparse_vector) > 0, "Should generate non-empty sparse vector for new term"

print("[OK] Temporary encoder can encode new terms during ingestion")

# After successful ingestion, commit
base3 = temp3  # In real code: self.sparse_encoder = temp_encoder
print(f"After commit, base encoder vocab: {list(base3.vocab.keys())}")
assert 'new' in base3.vocab, "New term should now be in base encoder"

# Cleanup
Path(encoder_path).unlink()

print("\n=== All Transactional Tests Passed ===")
