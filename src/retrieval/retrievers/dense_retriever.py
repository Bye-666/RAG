"""Dense vector retrieval."""

from typing import List, Dict, Any, Optional
from ...libs.vector_store.base import BaseVectorStore


class DenseRetriever:
    """Retrieve documents using dense vector similarity.
    
    Uses embedding-based dense vectors for semantic search.
    """

    def __init__(self, vector_store: BaseVectorStore):
        """Initialize DenseRetriever.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store

    def retrieve(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents using dense vector.
        
        Args:
            query_vector: Dense query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of retrieved documents with scores
        """
        results = self.vector_store.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
