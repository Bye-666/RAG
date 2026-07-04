#!/usr/bin/env python3
"""
Ingestion script for RAG system.

Usage:
    python scripts/ingest.py <file_or_directory> [options]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Settings
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.batch_processor import BatchProcessor
from src.ingestion.embedders.dense_encoder import DenseEncoder
from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.loader import ComponentLoader


def print_progress(stage: str, current: int, total: int):
    """Print progress bar."""
    if total == 0:
        percentage = 100
    else:
        percentage = int((current / total) * 100)
    
    bar_length = 40
    filled = int(bar_length * current / total) if total > 0 else bar_length
    bar = '=' * filled + '-' * (bar_length - filled)
    
    print(f"\r{stage:12s} [{bar}] {percentage:3d}% ({current}/{total})", end='', flush=True)
    
    if current == total:
        print()  # New line when complete


def collect_files(path: Path, extensions: list = None) -> list:
    """Collect files from path.
    
    Args:
        path: File or directory path
        extensions: File extensions to include (e.g., ['.pdf'])
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.pdf']
    
    if path.is_file():
        if path.suffix.lower() in extensions:
            return [path]
        else:
            print(f"Warning: {path} is not a supported file type")
            return []
    
    elif path.is_dir():
        files = []
        for ext in extensions:
            files.extend(path.glob(f"**/*{ext}"))
        return sorted(files)
    
    else:
        print(f"Error: {path} does not exist")
        return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest single file
  python scripts/ingest.py data/documents/report.pdf
  
  # Ingest all PDFs in directory
  python scripts/ingest.py data/documents/
  
  # Use custom config
  python scripts/ingest.py data/documents/ --config config/custom.yaml
"""
    )
    
    parser.add_argument(
        'path',
        type=str,
        help='File or directory to ingest'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Config file path (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.pdf'],
        help='File extensions to process (default: .pdf)'
    )
    
    args = parser.parse_args()
    
    # Validate path
    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    # Collect files
    print(f"Scanning for files in: {input_path}")
    files = collect_files(input_path, args.extensions)
    
    if not files:
        print("No files found to process")
        sys.exit(0)
    
    print(f"Found {len(files)} file(s) to process\n")
    
    # Load config
    try:
        config = Settings.from_yaml(args.config)
    except Exception as e:
        print(f"Warning: Could not load config from {args.config}: {e}")
        print("Using default configuration\n")
        config = Settings()
    
    # Initialize components
    print("Initializing pipeline components...")

    try:
        # Loader
        loader = PDFLoader()

        # LLM, Embedding, Splitter from config
        component_loader = ComponentLoader(config)
        llm = component_loader.get_llm()
        embedding = component_loader.get_embedding()
        splitter = component_loader.get_splitter()
        vector_store = component_loader.get_vector_store()
        
        # Encoders
        dense_encoder = DenseEncoder(embedding)
        
        # Load or train sparse encoder
        sparse_encoder_path = Path("data/db/bm25_encoder.pkl")
        if sparse_encoder_path.exists():
            print(f"Loading BM25 encoder from {sparse_encoder_path}")
            sparse_encoder = BM25SparseEncoder.load(sparse_encoder_path)

            # Incrementally update with new documents
            print("Updating BM25 encoder with new documents...")
            all_texts = []
            for file_path in files:
                try:
                    doc = loader.load(file_path)
                    chunks = splitter.split(doc.text)
                    all_texts.extend(chunks)
                except Exception as e:
                    print(f"Warning: Could not load {file_path} for BM25 update: {e}")

            if all_texts:
                sparse_encoder.partial_fit(all_texts)
                print(f"[OK] BM25 encoder updated with {len(all_texts)} text chunks")
                print(f"    Vocabulary size: {len(sparse_encoder.vocab)}, Total documents: {sparse_encoder.doc_count}")
        else:
            print("BM25 encoder not found, training on documents...")
            sparse_encoder = BM25SparseEncoder()
            sparse_encoder_path.parent.mkdir(parents=True, exist_ok=True)

            # Collect all text chunks from files for training
            all_texts = []
            for file_path in files:
                try:
                    doc = loader.load(file_path)
                    chunks = splitter.split(doc.text)
                    all_texts.extend(chunks)
                except Exception as e:
                    print(f"Warning: Could not load {file_path} for BM25 training: {e}")

            if all_texts:
                sparse_encoder.fit(all_texts)
                print(f"[OK] BM25 encoder trained on {len(all_texts)} text chunks")
            else:
                print("Warning: No texts available for BM25 training")

        # Batch processor and upserter
        batch_processor = BatchProcessor(dense_encoder, sparse_encoder)
        vector_upserter = VectorUpserter(vector_store)
        
        # Pipeline
        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            batch_processor=batch_processor,
            vector_upserter=vector_upserter,
            batch_size=args.batch_size
        )
        
        print("Pipeline initialized successfully\n")
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Process files
    print("Starting ingestion...\n")

    result = pipeline.ingest_files(files, progress_callback=print_progress)

    # Save BM25 encoder after successful ingestion
    if result['successful'] > 0:
        sparse_encoder.save(sparse_encoder_path)
        print(f"\n[OK] BM25 encoder saved to {sparse_encoder_path}")

    # Print summary
    print("\n" + "="*60)
    print("Ingestion Summary")
    print("="*60)
    print(f"Total files:      {result['total_files']}")
    print(f"Successful:       {result['successful']}")
    print(f"Failed:           {result['failed']}")
    print(f"Total chunks:     {result['total_chunks']}")
    print("="*60)
    
    # Show failures if any
    if result['failed'] > 0:
        print("\nFailed files:")
        for r in result['results']:
            if not r['success']:
                print(f"  - {r['file']}: {r.get('error', 'Unknown error')}")
    
    print("\nIngestion complete!")
    
    sys.exit(0 if result['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
