# BM25 Sparse Encoder Tests

This directory contains unit tests for the BM25 sparse encoder implementation, focusing on incremental updates, deduplication, and transactional operations.

## Test Files

### test_bm25.py
Tests basic BM25 encoding with Chinese (jieba) tokenization.
- Verifies vocabulary loading
- Tests encoding of Chinese queries
- Validates sparse vector generation

### test_partial_fit.py
Tests incremental update functionality (`partial_fit()`).
- Initial training with first batch of documents
- Incremental updates with new documents
- Verifies vocabulary expansion
- Tests encoding of newly added terms

### test_deduplication.py
Tests document deduplication using content hashing.
- Verifies duplicate documents are skipped
- Tests hash-based deduplication
- Validates document count after deduplication

### test_save_load.py
Tests persistence of incremental stats.
- Saves encoder with `term_doc_freq`, `total_doc_length`, `doc_hashes`
- Loads encoder and verifies stats are restored
- Tests incremental updates after load

### test_migration_transaction.py
Tests old pickle format migration and transactional updates.
- Detects old format pickles (without incremental stats)
- Validates migration warnings
- Tests that `partial_fit()` correctly rejects old format encoders

### test_transactional.py
Tests transactional BM25 updates using the clone strategy.
- Verifies `clone()` creates independent copies
- Tests commit/rollback behavior
- Validates that temporary encoders can encode new terms during ingestion
- Ensures failed ingestion doesn't corrupt saved encoder

## Running Tests

Run individual tests:
```bash
python tests/unit/bm25/test_bm25.py
python tests/unit/bm25/test_partial_fit.py
python tests/unit/bm25/test_deduplication.py
python tests/unit/bm25/test_save_load.py
python tests/unit/bm25/test_migration_transaction.py
python tests/unit/bm25/test_transactional.py
```

Run all BM25 tests:
```bash
python -m pytest tests/unit/bm25/
```

## Key Fixes Validated

These tests validate the following critical fixes:

1. **Incremental Updates**: `partial_fit()` correctly updates vocabulary without full retraining
2. **Deduplication**: Duplicate documents are detected via content hashing
3. **Transactional Updates**: Failed ingestion doesn't corrupt the encoder state
4. **Old Format Migration**: Legacy pickles are detected and users are prompted to retrain
5. **Chinese Tokenization**: Jieba correctly tokenizes Chinese text for sparse retrieval
