"""Ingestion Management Page - Manage document ingestion."""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.session_init import init_session_state
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.batch_processor import BatchProcessor
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder
from src.trace.trace_context import get_trace_recorder

st.set_page_config(page_title="Ingestion 管理", page_icon="📥", layout="wide")

# Initialize session state
init_session_state()

st.title("📥 Ingestion 管理")
st.markdown("管理文档摄取流程")

st.markdown("---")

# File uploader
st.subheader("📤 上传文档")

uploaded_file = st.file_uploader(
    "选择 PDF 文件",
    type=["pdf"],
    help="上传 PDF 文档进行摄取"
)

col1, col2 = st.columns(2)

with col1:
    enable_transform = st.checkbox("启用 Transform 处理", value=False, help="包括文本优化、元数据提取等")

with col2:
    batch_size = st.slider("批处理大小", min_value=8, max_value=64, value=32)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = Path("data/temp") / uploaded_file.name
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"文件已上传: {uploaded_file.name}")

    if st.button("🚀 开始摄取", type="primary"):
        # Initialize pipeline
        try:
            component_loader = st.session_state.component_loader

            # Load BM25 encoder or create new
            sparse_encoder_path = Path("data/db/bm25_encoder.pkl")
            if sparse_encoder_path.exists():
                sparse_encoder = BM25SparseEncoder.load(sparse_encoder_path)
                st.info("✓ 使用已有的 BM25 编码器")
            else:
                sparse_encoder = BM25SparseEncoder()
                sparse_encoder_path.parent.mkdir(parents=True, exist_ok=True)
                st.info("→ 将在摄取后训练 BM25 编码器")

            # First pass: Load and split to train BM25 if needed
            if not sparse_encoder.vocab:
                with st.spinner("准备训练 BM25 编码器..."):
                    # Load and split document to get text corpus
                    loader = PDFLoader()
                    document = loader.load(temp_path)
                    splitter = component_loader.get_splitter()
                    text_chunks = splitter.split(document.text)

                    # Train BM25 on the corpus
                    sparse_encoder.fit(text_chunks)
                    st.success(f"✓ BM25 编码器已训练（词汇量: {len(sparse_encoder.vocab)}）")

            # Create pipeline
            pipeline = IngestionPipeline(
                loader=PDFLoader(),
                splitter=component_loader.get_splitter(),
                batch_processor=BatchProcessor(
                    dense_encoder=component_loader.get_embedding(),
                    sparse_encoder=sparse_encoder
                ),
                vector_upserter=VectorUpserter(component_loader.get_vector_store()),
                batch_size=batch_size,
                enable_transform=enable_transform
            )

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(stage, current, total):
                if total > 0:
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"{stage}: {current}/{total}")

            # Run ingestion
            with st.spinner("正在摄取文档..."):
                result = pipeline.ingest_file(
                    temp_path,
                    progress_callback=progress_callback,
                    enable_trace=True
                )

            # Save BM25 encoder
            sparse_encoder.save(sparse_encoder_path)

            # Display results
            if result["success"]:
                st.success("✅ 摄取成功!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("处理的数据块", result["chunks_processed"])
                with col2:
                    st.metric("上传的数据块", result["chunks_uploaded"])

                # Show trace
                traces = get_trace_recorder().get_traces(trace_type="ingestion", limit=1)
                if traces:
                    with st.expander("查看追踪详情"):
                        st.json(traces[0])
            else:
                st.error(f"❌ 摄取失败: {result.get('error', '未知错误')}")

        except Exception as e:
            st.error(f"摄取过程出错: {str(e)}")
            st.code(str(e))

        finally:
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()

st.markdown("---")

# Batch ingestion
st.subheader("📦 批量摄取")

data_dir = st.text_input("数据目录", value="data/documents", help="包含 PDF 文件的目录")

if st.button("📂 扫描目录"):
    data_path = Path(data_dir)
    if data_path.exists():
        pdf_files = list(data_path.glob("*.pdf"))
        st.write(f"找到 {len(pdf_files)} 个 PDF 文件")

        if pdf_files:
            st.dataframe([{"文件名": f.name, "大小": f"{f.stat().st_size / 1024:.1f} KB"} for f in pdf_files])

            if st.button("🚀 批量摄取全部", type="primary"):
                st.info("批量摄取功能开发中...")
    else:
        st.warning(f"目录不存在: {data_dir}")

st.markdown("---")

# Recent ingestions
st.subheader("📋 最近的摄取记录")

ingestion_traces = get_trace_recorder().get_traces(trace_type="ingestion", limit=10)

if ingestion_traces:
    for i, trace in enumerate(reversed(ingestion_traces)):
        with st.expander(f"摄取 {i+1} - {trace.get('operation', 'N/A')}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**耗时**: {trace.get('duration_ms', 0):.2f} ms")

            with col2:
                status = "成功" if trace.get('error') is None else "失败"
                st.write(f"**状态**: {status}")

            with col3:
                st.write(f"**时间**: {trace.get('start_time', 'N/A')[:19]}")

            if trace.get('steps'):
                st.write("**步骤详情**:")
                for step in trace['steps']:
                    st.text(f"  - {step['name']}: {step.get('data', {})}")

            if trace.get('error'):
                st.error(f"错误: {trace['error']}")
else:
    st.info("暂无摄取记录")
