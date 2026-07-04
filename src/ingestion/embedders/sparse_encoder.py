"""BM25 sparse vector encoding."""

import math
import pickle
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any


class BM25SparseEncoder:
    """BM25 sparse vector encoder.
    
    Converts traditional BM25 algorithm output into sparse vector format
    compatible with Milvus sparse vector index.
    
    BM25 formula: score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    where:
    - idf: inverse document frequency
    - tf: term frequency in document
    - dl: document length
    - avgdl: average document length
    - k1: term frequency saturation parameter (default: 1.5)
    - b: length normalization parameter (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25SparseEncoder.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0-1)
        """
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}  # term -> term_id mapping
        self.idf: Dict[int, float] = {}  # term_id -> idf value
        self.avgdl: float = 0  # average document length
        self.doc_count: int = 0  # total number of documents
        self.term_doc_freq: Dict[str, int] = {}  # term -> document frequency (for incremental updates)
        self.total_doc_length: float = 0  # sum of all document lengths (for avgdl calculation)

    def fit(self, documents: List[str]):
        """Train on document collection to build vocabulary and IDF.

        Args:
            documents: List of text documents
        """
        if not documents:
            raise ValueError("Cannot fit on empty document list")

        doc_lengths = []
        term_doc_freq = Counter()  # count documents containing each term

        # Count term frequencies and document lengths
        for doc in documents:
            terms = self._tokenize(doc)
            doc_lengths.append(len(terms))
            unique_terms = set(terms)

            # Assign ID to new terms
            for term in unique_terms:
                if term not in self.vocab:
                    self.vocab[term] = len(self.vocab)
                term_doc_freq[term] += 1

        self.doc_count = len(documents)
        self.avgdl = sum(doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        self.total_doc_length = sum(doc_lengths)
        self.term_doc_freq = dict(term_doc_freq)

        # Calculate IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, term_id in self.vocab.items():
            df = term_doc_freq[term]
            idf_value = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            self.idf[term_id] = idf_value

    def partial_fit(self, documents: List[str]):
        """Incrementally update vocabulary and IDF with new documents.

        This method allows adding new documents without retraining from scratch.
        New terms are added to vocabulary, and IDF values are recalculated for all terms.

        Args:
            documents: List of new text documents to add
        """
        if not documents:
            return

        # Process new documents
        new_doc_lengths = []
        new_term_doc_freq = Counter()

        for doc in documents:
            terms = self._tokenize(doc)
            new_doc_lengths.append(len(terms))
            unique_terms = set(terms)

            # Add new terms to vocabulary and update document frequency
            for term in unique_terms:
                if term not in self.vocab:
                    self.vocab[term] = len(self.vocab)
                new_term_doc_freq[term] += 1

        # Update statistics
        self.doc_count += len(documents)
        self.total_doc_length += sum(new_doc_lengths)
        self.avgdl = self.total_doc_length / self.doc_count if self.doc_count > 0 else 0

        # Merge new term document frequencies with existing ones
        for term, df in new_term_doc_freq.items():
            self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + df

        # Recalculate IDF for all terms with updated document count
        for term, term_id in self.vocab.items():
            df = self.term_doc_freq.get(term, 0)
            idf_value = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            self.idf[term_id] = idf_value

    def encode(self, text: str) -> Dict[int, float]:
        """Encode text into sparse vector.

        Args:
            text: Text to encode

        Returns:
            Sparse vector in Milvus Lite format: {term_id: score, ...}
        """
        if not self.vocab:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        terms = self._tokenize(text)
        term_freq = Counter(terms)
        doc_len = len(terms)

        sparse_vector = {}

        for term, tf in term_freq.items():
            if term in self.vocab:
                term_id = self.vocab[term]
                idf_value = self.idf[term_id]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = idf_value * (numerator / denominator)

                # Milvus Lite 格式：dict[int, float]
                sparse_vector[int(term_id)] = float(score)

        return sparse_vector

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms.

        Uses jieba for Chinese text segmentation.
        Falls back to whitespace splitting for non-Chinese text.

        Args:
            text: Text to tokenize

        Returns:
            List of terms
        """
        # 尝试使用 jieba 进行中文分词
        try:
            import jieba
            # jieba 分词，返回词语列表
            terms = list(jieba.cut(text))
            # 过滤空字符串和单字符（保留多字词）
            terms = [term.strip() for term in terms if term.strip() and len(term.strip()) > 1]
            return terms
        except ImportError:
            # jieba 未安装，使用空格分词
            print("[WARNING] jieba not installed, using whitespace tokenization. Install with: pip install jieba")
            return text.split()

    def save(self, path: str | Path):
        """Persist vocabulary and IDF to disk.
        
        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
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
    def load(cls, path: str | Path) -> 'BM25SparseEncoder':
        """Load trained encoder from disk.
        
        Args:
            path: Path to saved encoder file
            
        Returns:
            Loaded encoder instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Encoder file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        encoder = cls(k1=data['k1'], b=data['b'])
        encoder.vocab = data['vocab']
        encoder.idf = data['idf']
        encoder.avgdl = data['avgdl']
        encoder.doc_count = data['doc_count']
        return encoder
