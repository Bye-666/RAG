"""Ingestion Tracing Page - View and analyze ingestion traces."""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Optional plotly import
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.session_init import init_session_state
from src.trace.trace_context import get_trace_recorder

st.set_page_config(page_title="Ingestion 追踪", page_icon="📊", layout="wide")

# Initialize session state
init_session_state()

st.title("📊 Ingestion 追踪")
st.markdown("文档摄取流程追踪和性能分析")

st.markdown("---")

# Get ingestion traces
ingestion_traces = get_trace_recorder().get_traces(trace_type="ingestion")

if ingestion_traces:
    # Summary statistics
    st.subheader("📈 摄取统计")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总摄取次数", len(ingestion_traces))

    with col2:
        successful = sum(1 for t in ingestion_traces if t.get('error') is None)
        st.metric("成功次数", successful)

    with col3:
        failed = len(ingestion_traces) - successful
        st.metric("失败次数", failed)

    with col4:
        success_rate = (successful / len(ingestion_traces)) * 100 if ingestion_traces else 0
        st.metric("成功率", f"{success_rate:.1f}%")

    st.markdown("---")

    # Performance analysis
    st.subheader("⏱️ 性能分析")

    durations = [t.get('duration_ms', 0) for t in ingestion_traces if t.get('duration_ms')]

    if durations:
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_duration = sum(durations) / len(durations)
            st.metric("平均耗时", f"{avg_duration:.2f} ms")

        with col2:
            st.metric("最快摄取", f"{min(durations):.2f} ms")

        with col3:
            st.metric("最慢摄取", f"{max(durations):.2f} ms")

        # Duration chart
        st.markdown("**摄取耗时趋势:**")
        df = pd.DataFrame({
            "序号": list(range(1, len(durations) + 1)),
            "耗时 (ms)": durations
        })

        if HAS_PLOTLY:
            fig = px.line(df, x="序号", y="耗时 (ms)", title="摄取耗时趋势")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df.set_index("序号"))
            st.info("提示: 安装 plotly 可获得更好的图表体验 (pip install plotly)")

    st.markdown("---")

    # Step breakdown
    st.subheader("🔍 步骤分析")

    # Analyze step durations across all traces
    step_stats = {}

    for trace in ingestion_traces:
        steps = trace.get('steps', [])
        if len(steps) > 1:
            for i in range(len(steps) - 1):
                step_name = steps[i]['name']
                # Calculate time between steps (simplified)
                if step_name not in step_stats:
                    step_stats[step_name] = []
                step_stats[step_name].append(1)  # Placeholder

    if step_stats:
        step_data = []
        for step_name, counts in step_stats.items():
            step_data.append({
                "步骤": step_name,
                "执行次数": len(counts)
            })

        df_steps = pd.DataFrame(step_data)
        st.dataframe(df_steps, use_container_width=True)

    st.markdown("---")

    # Detailed trace list
    st.subheader("📋 详细追踪记录")

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        show_all = st.checkbox("显示所有记录", value=False)

    with col2:
        show_errors_only = st.checkbox("仅显示错误", value=False)

    # Filter traces
    filtered_traces = ingestion_traces

    if show_errors_only:
        filtered_traces = [t for t in filtered_traces if t.get('error') is not None]

    if not show_all:
        filtered_traces = filtered_traces[-20:]  # Show last 20

    st.write(f"**显示 {len(filtered_traces)} 条记录**")

    # Display traces
    for i, trace in enumerate(reversed(filtered_traces)):
        operation = trace.get('operation', 'N/A')
        duration = trace.get('duration_ms', 0)
        has_error = trace.get('error') is not None

        # Color code by status
        status_emoji = "❌" if has_error else "✅"

        with st.expander(f"{status_emoji} 摄取 {i+1} - {operation} ({duration:.2f} ms)"):
            # Basic info
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Trace ID**: {trace.get('trace_id', 'N/A')[:16]}...")

            with col2:
                st.write(f"**开始时间**: {trace.get('start_time', 'N/A')[:19]}")

            with col3:
                st.write(f"**耗时**: {duration:.2f} ms")

            # Steps
            st.write("**执行步骤:**")
            steps = trace.get('steps', [])

            if steps:
                for step in steps:
                    st.text(f"  • {step['name']}")
                    data = step.get('data', {})
                    if data:
                        for key, value in data.items():
                            st.text(f"      - {key}: {value}")

            # Result
            result = trace.get('result')
            if result:
                st.write("**结果:**")
                st.json(result)

            # Error
            if has_error:
                st.error(f"**错误**: {trace['error']}")

            # Full trace data
            with st.expander("查看完整追踪数据 (JSON)"):
                st.json(trace)

    st.markdown("---")

    # Export functionality
    st.subheader("💾 导出数据")

    if st.button("导出所有追踪记录为 JSON"):
        import json
        from datetime import datetime

        filename = f"ingestion_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_data = json.dumps(ingestion_traces, indent=2, ensure_ascii=False)

        st.download_button(
            label="下载 JSON 文件",
            data=json_data,
            file_name=filename,
            mime="application/json"
        )

else:
    st.info("暂无摄取追踪记录")
    st.markdown("""
    ### 如何生成追踪记录？

    1. 前往 **Ingestion 管理** 页面
    2. 上传并摄取一个 PDF 文档
    3. 返回此页面查看追踪详情

    追踪记录将包含：
    - 执行耗时
    - 各步骤详情
    - 数据块数量
    - 错误信息（如有）
    """)
