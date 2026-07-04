"""System Settings Page - Configuration and management."""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.session_init import init_session_state
from src.trace.trace_context import get_trace_recorder

st.set_page_config(page_title="系统设置", page_icon="⚙️", layout="wide")

# Initialize session state
init_session_state()

st.title("⚙️ 系统设置")
st.markdown("系统配置和管理")

st.markdown("---")

# Configuration display
st.subheader("📋 当前配置")

config = st.session_state.config

tab1, tab2, tab3, tab4 = st.tabs(["LLM", "Embedding", "向量存储", "其他"])

with tab1:
    st.markdown("### LLM 配置")
    api_key = config.get("llm.api_key", "")
    st.json({
        "provider": config.get("llm.provider", "dashscope"),
        "model": config.get("llm.model", "qwen-max"),
        "base_url": config.get("llm.base_url", ""),
        "api_key": "***" + api_key[-4:] if api_key else "未配置",
        "temperature": config.get("llm.temperature", 0.7),
        "max_tokens": config.get("llm.max_tokens", 2000)
    })

with tab2:
    st.markdown("### Embedding 配置")
    api_key = config.get("embedding.api_key", "")
    st.json({
        "provider": config.get("embedding.provider", "dashscope"),
        "model": config.get("embedding.model", "text-embedding-v4"),
        "base_url": config.get("embedding.base_url", ""),
        "api_key": "***" + api_key[-4:] if api_key else "未配置",
        "dimension": config.get("embedding.dimension", 2048)
    })

with tab3:
    st.markdown("### 向量存储配置")
    st.json({
        "uri": config.get("milvus.uri", "milvus.db"),
        "collection_name": config.get("milvus.collection_name", "rag_knowledge_hub"),
        "dense_dim": config.get("milvus.dense_dim", 2048)
    })

with tab4:
    st.markdown("### 其他配置")
    st.json({
        "log_level": config.get("log_level", "INFO"),
        "log_dir": config.get("log_dir", "logs")
    })

st.markdown("---")

# System status
st.subheader("💚 系统状态")

col1, col2, col3 = st.columns(3)

with col1:
    # Test vector store connection
    try:
        vector_store = st.session_state.component_loader.get_vector_store()
        stats = vector_store.get_collection_stats()
        st.success("✅ 向量存储: 连接正常")
        st.write(f"数据块数: {stats.get('row_count', 0)}")
    except Exception as e:
        st.error("❌ 向量存储: 连接失败")
        st.code(str(e))

with col2:
    # Test embedding
    try:
        embedding = st.session_state.component_loader.get_embedding()
        test_vec = embedding.embed("测试")
        st.success("✅ Embedding: 正常")
        st.write(f"维度: {len(test_vec)}")
    except Exception as e:
        st.error("❌ Embedding: 失败")
        st.code(str(e))

with col3:
    # Check BM25 encoder
    sparse_encoder_path = Path("data/db/bm25_encoder.pkl")
    if sparse_encoder_path.exists():
        st.success("✅ BM25 编码器: 已训练")
    else:
        st.warning("⚠️ BM25 编码器: 未训练")

st.markdown("---")

# Data management
st.subheader("🗂️ 数据管理")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 清除追踪数据")
    st.write("清除所有追踪记录（不影响向量库数据）")

    if st.button("🗑️ 清除追踪记录", type="secondary"):
        get_trace_recorder().clear()
        st.success("✅ 追踪记录已清除")
        st.rerun()

with col2:
    st.markdown("### 重置 BM25 编码器")
    st.write("删除 BM25 编码器（需要重新摄取文档）")

    if st.button("🔄 重置 BM25", type="secondary"):
        sparse_encoder_path = Path("data/db/bm25_encoder.pkl")
        if sparse_encoder_path.exists():
            sparse_encoder_path.unlink()
            st.success("✅ BM25 编码器已重置")
            st.rerun()
        else:
            st.info("BM25 编码器不存在")

st.markdown("---")

# Dangerous operations
st.subheader("⚠️ 危险操作")

with st.expander("清除向量库数据"):
    st.warning("⚠️ 此操作将删除向量库中的所有数据，无法恢复！")

    confirm = st.text_input("输入 'DELETE' 确认删除", key="delete_confirm")

    if st.button("删除所有向量数据", type="secondary") and confirm == "DELETE":
        try:
            vector_store = st.session_state.component_loader.get_vector_store()
            collection_name = config.get("milvus.collection_name", "rag_knowledge_hub")
            # Drop and recreate collection
            vector_store.client.drop_collection(collection_name)
            st.success("✅ 向量数据已清除")
            st.info("请重新摄取文档")
            st.rerun()
        except Exception as e:
            st.error(f"删除失败: {str(e)}")

st.markdown("---")

# System info
st.subheader("ℹ️ 系统信息")

import sys as python_sys
import platform

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Python 环境**")
    st.write(f"版本: {python_sys.version}")
    st.write(f"平台: {platform.platform()}")

with col2:
    st.markdown("**依赖库**")
    try:
        import pymilvus
        st.write(f"pymilvus: {pymilvus.__version__}")
    except:
        st.write("pymilvus: 未安装")

    try:
        import streamlit
        st.write(f"streamlit: {streamlit.__version__}")
    except:
        st.write("streamlit: 未安装")

st.markdown("---")

# Logs
st.subheader("📄 日志查看")

log_dir = Path(config.get("log_dir", "logs"))

if log_dir.exists():
    log_files = list(log_dir.glob("*.jsonl")) + list(log_dir.glob("*.log"))

    if log_files:
        selected_log = st.selectbox("选择日志文件", log_files)

        if selected_log:
            num_lines = st.slider("显示行数", 10, 200, 50)

            with open(selected_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                last_lines = lines[-num_lines:]

            st.code("".join(last_lines), language="log")
    else:
        st.info("暂无日志文件")
else:
    st.info(f"日志目录不存在: {log_dir}")
