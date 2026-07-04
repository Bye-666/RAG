"""Query Tracing Page - View and analyze query traces."""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Optional plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.session_init import init_session_state
from src.retrieval.query_processor import QueryProcessor
from src.trace.trace_context import get_trace_recorder
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.retrievers.dense_retriever import DenseRetriever
from src.retrieval.retrievers.sparse_retriever import SparseRetriever
from src.ingestion.embedders.sparse_encoder import BM25SparseEncoder

st.set_page_config(page_title="Query 追踪", page_icon="🔍", layout="wide")

# Initialize session state
init_session_state()

st.title("🔍 Query 追踪")
st.markdown("查询执行追踪和性能分析")

st.markdown("---")

# Live query test
st.subheader("🎯 实时查询测试")

query_text = st.text_input("输入查询", placeholder="例如: 什么是机器学习？")

col1, col2, col3 = st.columns(3)

with col1:
    top_k = st.slider("返回结果数", min_value=1, max_value=20, value=10)

with col2:
    enable_rerank = st.checkbox("启用重排序", value=False)

with col3:
    enable_trace = st.checkbox("启用追踪", value=True)

if st.button("🚀 执行查询", type="primary") and query_text:
    try:
        # Load BM25 encoder
        sparse_encoder_path = Path("data/db/bm25_encoder.pkl")
        if not sparse_encoder_path.exists():
            st.error("BM25 编码器未找到，请先进行文档摄取")
            st.stop()

        sparse_encoder = BM25SparseEncoder.load(sparse_encoder_path)

        # Initialize components
        component_loader = st.session_state.component_loader
        embedding = component_loader.get_embedding()
        vector_store = component_loader.get_vector_store()

        # Query processor
        query_processor = QueryProcessor(None, embedding, sparse_encoder)

        # Retrievers
        dense_retriever = DenseRetriever(vector_store)
        sparse_retriever = SparseRetriever(vector_store)
        hybrid_search = HybridSearch(dense_retriever, sparse_retriever)

        # Process query
        with st.spinner("正在处理查询..."):
            processed = query_processor.process(query_text, rewrite=False)

            # Search
            results = hybrid_search.search(
                dense_vector=processed["dense_vector"],
                sparse_vector=processed["sparse_vector"],
                top_k=top_k,
                enable_trace=enable_trace
            )

        # Display results
        st.success(f"✅ 找到 {len(results)} 个结果")

        for i, result in enumerate(results):
            with st.expander(f"结果 {i+1} - 相关度: {result.get('rrf_score', 0):.4f}"):
                st.write("**文本片段:**")
                st.text(result.get("text", "")[:500])

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID**: {result.get('id', 'N/A')}")
                with col2:
                    st.write(f"**RRF 分数**: {result.get('rrf_score', 0):.4f}")

                st.write("**元数据:**")
                st.json(result.get("metadata", {}))

        # Show trace
        if enable_trace:
            st.markdown("---")
            st.subheader("📊 执行追踪")

            traces = get_trace_recorder().get_traces(trace_type="query", limit=1)
            if traces:
                trace = traces[0]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总耗时", f"{trace.get('duration_ms', 0):.2f} ms")
                with col2:
                    st.metric("执行步骤", len(trace.get('steps', [])))
                with col3:
                    st.metric("最终结果", trace.get('result', {}).get('final_result_count', 0))

                # Step timeline
                st.write("**执行步骤时间线:**")
                steps = trace.get('steps', [])
                if steps:
                    step_data = []
                    for step in steps:
                        step_data.append({
                            "步骤": step['name'],
                            "数据": str(step.get('data', {}))
                        })
                    st.dataframe(step_data, use_container_width=True)

                with st.expander("查看完整追踪数据"):
                    st.json(trace)

    except Exception as e:
        st.error(f"查询失败: {str(e)}")
        st.code(str(e))

st.markdown("---")

# Historical query analysis
st.subheader("📈 历史查询分析")

query_traces = get_trace_recorder().get_traces(trace_type="query")

if query_traces:
    st.write(f"**总查询次数**: {len(query_traces)}")

    # Statistics
    durations = [t.get('duration_ms', 0) for t in query_traces if t.get('duration_ms')]
    if durations:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("平均耗时", f"{sum(durations)/len(durations):.2f} ms")
        with col2:
            st.metric("最快查询", f"{min(durations):.2f} ms")
        with col3:
            st.metric("最慢查询", f"{max(durations):.2f} ms")
        with col4:
            success_rate = sum(1 for t in query_traces if t.get('error') is None) / len(query_traces) * 100
            st.metric("成功率", f"{success_rate:.1f}%")

        # Duration chart
        st.markdown("**查询耗时趋势:**")
        df = pd.DataFrame({
            "序号": list(range(1, len(durations) + 1)),
            "耗时 (ms)": durations
        })

        if HAS_PLOTLY:
            fig = px.line(df, x="序号", y="耗时 (ms)", title="查询耗时趋势")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df.set_index("序号"))
            st.info("提示: 安装 plotly 可获得更好的图表体验 (pip install plotly)")

    # Recent queries
    st.markdown("---")
    st.subheader("📋 最近的查询")

    recent_traces = query_traces[-10:]
    for i, trace in enumerate(reversed(recent_traces)):
        with st.expander(f"查询 {i+1} - {trace.get('operation', 'N/A')}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**耗时**: {trace.get('duration_ms', 0):.2f} ms")
                st.write(f"**状态**: {'成功' if trace.get('error') is None else '失败'}")

            with col2:
                st.write(f"**时间**: {trace.get('start_time', 'N/A')[:19]}")
                result_count = trace.get('result', {}).get('final_result_count', 0)
                st.write(f"**结果数**: {result_count}")

            if trace.get('steps'):
                step_names = [s['name'] for s in trace['steps']]
                st.text(f"步骤: {' → '.join(step_names)}")

            if trace.get('error'):
                st.error(f"错误: {trace['error']}")
else:
    st.info("暂无查询记录，请执行一次查询")
