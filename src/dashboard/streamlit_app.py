"""Streamlit Dashboard for RAG System.

Main entry point for the multi-page dashboard application.

Usage:
    streamlit run src/dashboard/streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Settings
from src.libs.loader import ComponentLoader
from src.trace.trace_context import get_trace_recorder

# Page configuration
st.set_page_config(
    page_title="RAG System Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'config' not in st.session_state:
    try:
        st.session_state.config = Settings.from_yaml("config/settings.yaml")
    except Exception as e:
        st.error(f"⚠️ 配置文件加载失败: {str(e)}")
        st.info("使用默认配置")
        st.session_state.config = Settings()

if 'component_loader' not in st.session_state:
    try:
        st.session_state.component_loader = ComponentLoader(st.session_state.config)
    except Exception as e:
        st.warning(f"⚠️ 组件加载器初始化失败: {str(e)}")
        st.session_state.component_loader = None

if 'trace_recorder' not in st.session_state:
    # Initialize trace recorder in session state
    get_trace_recorder()

# Check dependencies
def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    try:
        import pymilvus
    except ImportError:
        missing.append("pymilvus")

    try:
        import streamlit
    except ImportError:
        missing.append("streamlit")

    return missing

missing_deps = check_dependencies()
if missing_deps:
    st.sidebar.error("⚠️ 缺少依赖")
    st.sidebar.code(f"pip install {' '.join(missing_deps)}")

# Sidebar navigation
st.sidebar.title("🎯 RAG System Dashboard")
st.sidebar.markdown("---")

# Main page content
st.title("📊 系统总览")
st.markdown("### 欢迎使用 RAG 知识库管理系统")

# System status
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="向量存储",
        value="Milvus",
        delta="运行中"
    )

with col2:
    st.metric(
        label="LLM 模型",
        value=st.session_state.config.get("llm.model", "未配置"),
        delta="就绪"
    )

with col3:
    st.metric(
        label="Embedding 模型",
        value=st.session_state.config.get("embedding.model", "未配置"),
        delta="就绪"
    )

with col4:
    # Get trace count
    traces = get_trace_recorder().get_traces()
    st.metric(
        label="追踪记录",
        value=len(traces),
        delta="条"
    )

st.markdown("---")

# Quick stats
st.subheader("📈 快速统计")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔍 查询统计")
    query_traces = get_trace_recorder().get_traces(trace_type="query")
    st.write(f"总查询次数: **{len(query_traces)}**")

    if query_traces:
        avg_duration = sum(t['duration_ms'] for t in query_traces if t.get('duration_ms')) / len(query_traces) if query_traces else 0
        st.write(f"平均耗时: **{avg_duration:.2f} ms**")

with col2:
    st.markdown("#### 📥 摄取统计")
    ingestion_traces = get_trace_recorder().get_traces(trace_type="ingestion")
    st.write(f"总摄取次数: **{len(ingestion_traces)}**")

    if ingestion_traces:
        successful = sum(1 for t in ingestion_traces if t.get('error') is None)
        st.write(f"成功率: **{successful/len(ingestion_traces)*100:.1f}%**")

st.markdown("---")

# Configuration display
st.subheader("⚙️ 系统配置")

config_col1, config_col2 = st.columns(2)

with config_col1:
    st.markdown("**向量存储配置**")
    vector_store_config = st.session_state.config.get("vector_store", {})
    st.json({
        "uri": vector_store_config.get("uri", "未配置"),
        "collection": vector_store_config.get("collection_name", "未配置"),
        "dense_dim": vector_store_config.get("dense_dim", 0)
    })

with config_col2:
    st.markdown("**模型配置**")
    llm_config = st.session_state.config.get("llm", {})
    embedding_config = st.session_state.config.get("embedding", {})
    api_key = llm_config.get("api_key", "")
    st.json({
        "llm": llm_config.get("model", "未配置"),
        "embedding": embedding_config.get("model", "未配置"),
        "api_key": "***" + api_key[-4:] if api_key else "未配置"
    })

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>RAG Knowledge Hub Dashboard v1.0</p>
    <p>使用说明：请从左侧菜单选择功能页面</p>
</div>
""", unsafe_allow_html=True)
