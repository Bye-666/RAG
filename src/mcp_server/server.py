"""MCP Server for RAG system."""

from typing import List, Dict, Any, Optional
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Settings
from src.libs.loader import ComponentLoader
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.batch_processor import BatchProcessor
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.retrievers.dense_retriever import DenseRetriever
from src.retrieval.retrievers.sparse_retriever import SparseRetriever
from src.retrieval.reranker_module import RerankerModule


class MCPServer:
    """MCP Server exposing RAG tools to Claude Desktop.

    Provides tools for:
    - Document ingestion
    - Query and retrieval
    - System management
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP Server.

        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        if config_path:
            self.config = Settings.from_yaml(config_path)
        else:
            self.config = Settings()

        # Initialize components
        self._init_components()

        # Initialize MCP server
        self.server = Server("rag-mcp-server")
        self._register_tools()

    def _init_components(self):
        """Initialize RAG system components."""
        self.component_loader = ComponentLoader(self.config)

        # Core components
        self.llm = self.component_loader.get_llm()
        self.embedding = self.component_loader.get_embedding()
        self.vector_store = self.component_loader.get_vector_store()
        self.splitter = self.component_loader.get_splitter()
        self.reranker = self.component_loader.get_reranker()

        # Load or create BM25 encoder
        self.sparse_encoder_path = Path("data/db/bm25_encoder.pkl")
        if self.sparse_encoder_path.exists():
            self.sparse_encoder = BM25SparseEncoder.load(self.sparse_encoder_path)
        else:
            self.sparse_encoder = BM25SparseEncoder()
            self.sparse_encoder_path.parent.mkdir(parents=True, exist_ok=True)

        # Ingestion pipeline
        self.pdf_loader = PDFLoader()
        self.batch_processor = BatchProcessor(
            dense_encoder=self.embedding,
            sparse_encoder=self.sparse_encoder
        )
        self.vector_upserter = VectorUpserter(self.vector_store)

        self.ingestion_pipeline = IngestionPipeline(
            loader=self.pdf_loader,
            splitter=self.splitter,
            batch_processor=self.batch_processor,
            vector_upserter=self.vector_upserter,
            enable_transform=False  # Disable transform for speed (enable if needed)
        )

        # Query pipeline
        self.query_processor = QueryProcessor(
            llm=self.llm,
            embedding=self.embedding,
            sparse_encoder=self.sparse_encoder
        )

        self.dense_retriever = DenseRetriever(self.vector_store)
        self.sparse_retriever = SparseRetriever(self.vector_store)
        self.hybrid_search = HybridSearch(self.dense_retriever, self.sparse_retriever)
        self.reranker_module = RerankerModule(self.reranker)

    def _register_tools(self):
        """Register all MCP tools."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="ingest_document",
                    description="Ingest a document into the RAG system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the document file"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="query",
                    description="Query the RAG system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query text"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="list_documents",
                    description="List all ingested documents",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="query_knowledge_hub",
                    description="Query a specific knowledge collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection": {
                                "type": "string",
                                "description": "Collection name"
                            },
                            "query": {
                                "type": "string",
                                "description": "Query text"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results",
                                "default": 10
                            }
                        },
                        "required": ["collection", "query"]
                    }
                ),
                Tool(
                    name="list_collections",
                    description="List all available knowledge collections",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_document_summary",
                    description="Get summary of a specific document",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID"
                            }
                        },
                        "required": ["doc_id"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            
            if name == "ingest_document":
                return await self._handle_ingest(arguments)

            elif name == "query":
                return await self._handle_query(arguments)

            elif name == "list_documents":
                return await self._handle_list_documents(arguments)

            elif name == "query_knowledge_hub":
                return await self._handle_query_knowledge_hub(arguments)

            elif name == "list_collections":
                return await self._handle_list_collections(arguments)

            elif name == "get_document_summary":
                return await self._handle_get_document_summary(arguments)

            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]

    async def _handle_ingest(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle document ingestion.

        Args:
            args: Tool arguments

        Returns:
            Response content
        """
        file_path = args.get("file_path")

        try:
            # Validate file exists
            path = Path(file_path)
            if not path.exists():
                result = {
                    "status": "error",
                    "message": f"File not found: {file_path}"
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            # Load and split document
            document = self.pdf_loader.load(path)
            text_chunks = self.splitter.split(document.text)

            # Create temporary encoder for this ingestion
            temp_sparse_encoder = self.sparse_encoder.clone()

            # Update temporary encoder with new document
            if temp_sparse_encoder.vocab:
                # Incremental update
                try:
                    new_count = temp_sparse_encoder.partial_fit(text_chunks)
                    skipped = len(text_chunks) - new_count
                except RuntimeError as e:
                    # Old pickle format without incremental stats
                    result = {
                        "status": "error",
                        "message": f"Cannot ingest: {str(e)}",
                        "file_path": str(file_path),
                        "note": "Please delete data/db/bm25_encoder.pkl and re-ingest all documents"
                    }
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
            else:
                # Initial training
                temp_sparse_encoder.fit(text_chunks)
                new_count = len(text_chunks)
                skipped = 0

            # Create temporary pipeline with updated encoder
            from src.ingestion.batch_processor import BatchProcessor
            temp_batch_processor = BatchProcessor(
                dense_encoder=self.ingestion_pipeline.batch_processor.dense_encoder,
                sparse_encoder=temp_sparse_encoder
            )

            # Temporarily replace batch processor
            original_batch_processor = self.ingestion_pipeline.batch_processor
            self.ingestion_pipeline.batch_processor = temp_batch_processor

            # Ingest document
            ingest_result = self.ingestion_pipeline.ingest_file(file_path)

            # Restore original batch processor
            self.ingestion_pipeline.batch_processor = original_batch_processor

            if ingest_result["success"]:
                # Commit temporary encoder to persistent storage
                self.sparse_encoder = temp_sparse_encoder
                self.sparse_encoder.save(self.sparse_encoder_path)

                result = {
                    "status": "success",
                    "message": f"Document ingested successfully",
                    "file_path": str(file_path),
                    "chunks_processed": ingest_result["chunks_processed"],
                    "chunks_uploaded": ingest_result["chunks_uploaded"],
                    "bm25_new_chunks": new_count,
                    "bm25_duplicates_skipped": skipped,
                    "vocab_size": len(self.sparse_encoder.vocab),
                    "total_documents": self.sparse_encoder.doc_count
                }
            else:
                result = {
                    "status": "error",
                    "message": f"Ingestion failed: {ingest_result.get('error', 'Unknown error')}",
                    "file_path": str(file_path)
                }

        except Exception as e:
            result = {
                "status": "error",
                "message": f"Error during ingestion: {str(e)}",
                "file_path": str(file_path)
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_query(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle query.

        Args:
            args: Tool arguments

        Returns:
            Response content
        """
        query = args.get("query")
        top_k = args.get("top_k", 10)
        rerank = args.get("rerank", True)  # Enable reranking by default

        try:
            # Check if BM25 encoder is trained
            if not self.sparse_encoder_path.exists():
                result = {
                    "status": "error",
                    "message": "System not ready. Please ingest documents first.",
                    "query": query
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            # Process query
            processed = self.query_processor.process(query, rewrite=False)

            # Hybrid search
            search_results = self.hybrid_search.search(
                dense_vector=processed["dense_vector"],
                sparse_vector=processed["sparse_vector"],
                top_k=top_k * 2 if rerank else top_k  # Get more for reranking
            )

            # Rerank if enabled and reranker available
            if rerank and self.reranker:
                search_results = self.reranker_module.rerank(
                    query=query,
                    results=search_results,
                    top_k=top_k
                )
            else:
                search_results = search_results[:top_k]

            # Format results
            formatted_results = []
            for r in search_results:
                formatted_results.append({
                    "id": r.get("id"),
                    "text": r.get("text", "")[:500],  # Truncate for readability
                    "score": r.get("rrf_score") or r.get("rerank_score", 0.0),
                    "metadata": r.get("metadata", {})
                })

            result = {
                "status": "success",
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results
            }

        except Exception as e:
            result = {
                "status": "error",
                "message": f"Query failed: {str(e)}",
                "query": query
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_list_documents(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle document listing.

        Args:
            args: Tool arguments

        Returns:
            Response content
        """
        try:
            # Query vector store for unique documents
            # Get a sample of documents by querying with empty filter
            stats = self.vector_store.get_collection_stats()

            # Try to get unique doc_ids from vector store
            # This is a simple implementation - in production you'd want a proper document index
            unique_docs = {}

            # Sample query to get documents (limitation: we can't list all without querying)
            # Alternative: maintain a separate document registry
            collection_name = self.config.get("vector_store.collection_name", "rag_collection")
            dense_dim = self.config.get("vector_store.dense_dim", 2048)

            sample_results = self.vector_store.query(
                collection_name=collection_name,
                data=[[0.0] * dense_dim],  # Dummy query
                limit=100,
                output_fields=["id", "doc_id", "metadata"]
            )

            if sample_results and len(sample_results) > 0:
                for result in sample_results[0]:
                    doc_id = result.get("entity", {}).get("doc_id")
                    if doc_id and doc_id not in unique_docs:
                        metadata = result.get("entity", {}).get("metadata", {})
                        unique_docs[doc_id] = {
                            "doc_id": doc_id,
                            "file_path": metadata.get("file_path", "Unknown"),
                            "chunks": 1  # Will be updated
                        }
                    elif doc_id:
                        unique_docs[doc_id]["chunks"] += 1

            result = {
                "status": "success",
                "total": len(unique_docs),
                "total_chunks": stats.get("row_count", 0) if stats else 0,
                "documents": list(unique_docs.values())
            }

        except Exception as e:
            result = {
                "status": "error",
                "message": f"Failed to list documents: {str(e)}",
                "total": 0,
                "documents": []
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    def run(self):
        """Run the MCP server."""
        import asyncio
        from mcp.server.stdio import stdio_server
        
        async def main():
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        
        asyncio.run(main())

    async def _handle_query_knowledge_hub(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle knowledge hub query.

        Args:
            args: Tool arguments

        Returns:
            Response content
        """
        collection = args.get("collection")
        query = args.get("query")
        top_k = args.get("top_k", 10)

        try:
            # Process query
            processed = self.query_processor.process(query, rewrite=False)

            # Create filter for specific collection
            filter_dict = {"collection": collection} if collection != "default" else None

            # Hybrid search with filter
            search_results = self.hybrid_search.search(
                dense_vector=processed["dense_vector"],
                sparse_vector=processed["sparse_vector"],
                top_k=top_k,
                filter_dict=filter_dict
            )

            # Format results
            formatted_results = []
            for r in search_results:
                formatted_results.append({
                    "id": r.get("id"),
                    "text": r.get("text", "")[:500],
                    "score": r.get("rrf_score", 0.0),
                    "metadata": r.get("metadata", {})
                })

            result = {
                "status": "success",
                "collection": collection,
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results
            }

        except Exception as e:
            result = {
                "status": "error",
                "message": f"Query failed: {str(e)}",
                "collection": collection,
                "query": query
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_list_collections(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle list collections.

        Args:
            args: Tool arguments

        Returns:
            Response content
        """
        try:
            # In this implementation, we use a single collection but can simulate
            # multiple collections via metadata filtering
            # For now, return the main collection info

            stats = self.vector_store.get_collection_stats()
            collection_name = self.config.get("vector_store.collection_name", "rag_collection")

            collections = [
                {
                    "name": collection_name,
                    "doc_count": stats.get("row_count", 0) if stats else 0,
                    "description": "Main RAG knowledge base",
                    "last_updated": "2026-07-03"  # Would track this in production
                }
            ]

            result = {
                "status": "success",
                "total": len(collections),
                "collections": collections
            }

        except Exception as e:
            result = {
                "status": "error",
                "message": f"Failed to list collections: {str(e)}",
                "total": 0,
                "collections": []
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_get_document_summary(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle get document summary.

        Args:
            args: Tool arguments

        Returns:
            Response content
        """
        doc_id = args.get("doc_id")

        try:
            # Query vector store for chunks belonging to this document
            # Use metadata filtering to get chunks with matching doc_id
            collection_name = self.config.get("vector_store.collection_name", "rag_collection")
            dense_dim = self.config.get("vector_store.dense_dim", 2048)

            query_results = self.vector_store.query(
                collection_name=collection_name,
                data=[[0.0] * dense_dim],  # Dummy query
                limit=1000,  # Get many chunks to find all from this doc
                output_fields=["id", "doc_id", "text", "metadata"],
                filter=f'doc_id == "{doc_id}"'
            )

            if not query_results or len(query_results) == 0 or len(query_results[0]) == 0:
                result = {
                    "status": "error",
                    "message": f"Document not found: {doc_id}",
                    "doc_id": doc_id
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            # Extract document information from chunks
            chunks = query_results[0]
            total_chunks = len(chunks)
            first_chunk = chunks[0].get("entity", {})
            metadata = first_chunk.get("metadata", {})

            # Aggregate text from first few chunks for preview
            preview_text = ""
            for i, chunk in enumerate(chunks[:3]):
                chunk_text = chunk.get("entity", {}).get("text", "")
                preview_text += chunk_text + "\n\n"
                if len(preview_text) > 1000:
                    break

            result = {
                "status": "success",
                "doc_id": doc_id,
                "title": metadata.get("title", doc_id),
                "chunks": total_chunks,
                "file_path": metadata.get("file_path", "Unknown"),
                "preview": preview_text[:1000].strip(),
                "metadata": {
                    "pages": metadata.get("pages"),
                    "file_type": metadata.get("file_type", "pdf"),
                    "created_at": metadata.get("created_at", "Unknown")
                }
            }

        except Exception as e:
            result = {
                "status": "error",
                "message": f"Failed to get document summary: {str(e)}",
                "doc_id": doc_id
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]
