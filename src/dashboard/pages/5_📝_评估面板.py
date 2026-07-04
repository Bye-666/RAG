"""Evaluation Panel Page - RAG system evaluation."""

import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.session_init import init_session_state
from src.evaluation import EvalRunner, CompositeEvaluator, RagasEvaluator
from src.trace.trace_context import get_trace_recorder

st.set_page_config(page_title="评估面板", page_icon="📝", layout="wide")

# Initialize session state
init_session_state()

st.title("📝 评估面板")
st.markdown("RAG 系统质量评估与 Golden Test Set 管理")

# Initialize evaluators in session state
if 'eval_runner' not in st.session_state:
    try:
        # Try to initialize with RagasEvaluator
        llm = st.session_state.component_loader.get_llm() if st.session_state.component_loader else None
        embedding = st.session_state.component_loader.get_embedding() if st.session_state.component_loader else None

        ragas_eval = RagasEvaluator(llm=llm, embedding=embedding)
        composite = CompositeEvaluator(ragas_evaluator=ragas_eval)
        st.session_state.eval_runner = EvalRunner(composite_evaluator=composite)
        st.session_state.ragas_available = ragas_eval.is_available()
    except Exception as e:
        # Fallback to basic evaluator
        composite = CompositeEvaluator()
        st.session_state.eval_runner = EvalRunner(composite_evaluator=composite)
        st.session_state.ragas_available = False

st.markdown("---")

# Ragas integration status
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.ragas_available:
        st.success("✅ Ragas 可用")
    else:
        st.warning("⚠️ Ragas 降级模式")

with col2:
    query_traces = get_trace_recorder().get_traces(trace_type="query")
    st.metric("总查询次数", len(query_traces))

with col3:
    ingestion_traces = get_trace_recorder().get_traces(trace_type="ingestion")
    if ingestion_traces:
        success = sum(1 for t in ingestion_traces if t.get('error') is None)
        success_rate = success / len(ingestion_traces) * 100
        st.metric("摄取成功率", f"{success_rate:.1f}%")
    else:
        st.metric("摄取成功率", "N/A")

st.markdown("---")

# Tabs for different evaluation features
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Golden Test Set",
    "✍️ 单次评估",
    "📊 评估历史",
    "⚙️ 配置"
])

# Tab 1: Golden Test Set
with tab1:
    st.subheader("🎯 Golden Test Set 管理")

    st.markdown("""
    Golden Test Set 是预定义的测试查询集合，用于系统性评估和回归测试。
    """)

    # File selector
    data_dir = Path("data/test")
    data_dir.mkdir(parents=True, exist_ok=True)

    test_files = list(data_dir.glob("*.json"))

    col1, col2 = st.columns([3, 1])

    with col1:
        if test_files:
            selected_file = st.selectbox(
                "选择测试集",
                test_files,
                format_func=lambda x: x.name
            )
        else:
            selected_file = None
            st.info("📋 未找到测试集文件")

    with col2:
        if st.button("创建模板"):
            template_path = data_dir / f"test_set_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            st.session_state.eval_runner.create_test_set_template(str(template_path))
            st.success(f"✅ 模板已创建: {template_path.name}")
            st.rerun()

    # Load and display test set
    if selected_file:
        try:
            num_cases = st.session_state.eval_runner.load_test_set(str(selected_file))

            st.success(f"✅ 已加载 {num_cases} 个测试用例")

            # Display test cases
            with st.expander("查看测试用例"):
                for i, case in enumerate(st.session_state.eval_runner.test_data[:5]):  # Show first 5
                    st.markdown(f"**用例 {i+1}: {case.get('id', 'N/A')}**")
                    st.write(f"问题: {case.get('question', 'N/A')}")
                    st.write(f"答案: {case.get('answer', 'N/A')[:100]}...")

                if num_cases > 5:
                    st.info(f"还有 {num_cases - 5} 个用例未显示")

            # Run evaluation
            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                include_retrieval = st.checkbox("评估检索", value=True)

            with col2:
                include_generation = st.checkbox("评估生成", value=True)

            with col3:
                run_eval = st.button("🚀 运行评估", type="primary")

            if run_eval:
                with st.spinner("正在运行评估..."):
                    try:
                        results = st.session_state.eval_runner.run_evaluation(
                            include_retrieval=include_retrieval,
                            include_generation=include_generation
                        )

                        st.success("✅ 评估完成！")

                        # Display summary
                        st.subheader("📊 评估结果")

                        summary = results["summary"]

                        # Retrieval metrics
                        if "retrieval" in summary and summary["retrieval"]:
                            st.markdown("**检索指标**")
                            ret_col1, ret_col2, ret_col3 = st.columns(3)

                            with ret_col1:
                                st.metric("Precision", f"{summary['retrieval'].get('avg_precision', 0):.3f}")
                            with ret_col2:
                                st.metric("Recall", f"{summary['retrieval'].get('avg_recall', 0):.3f}")
                            with ret_col3:
                                st.metric("F1 Score", f"{summary['retrieval'].get('avg_f1', 0):.3f}")

                        # Generation metrics
                        if "generation" in summary and summary["generation"]:
                            st.markdown("**生成指标**")
                            gen_metrics = summary["generation"]

                            gen_cols = st.columns(min(len(gen_metrics), 4))
                            for i, (metric, value) in enumerate(gen_metrics.items()):
                                with gen_cols[i % len(gen_cols)]:
                                    st.metric(metric.replace("avg_", "").title(), f"{value:.3f}")

                        # Overall score
                        if "overall" in summary and summary["overall"]:
                            st.markdown("**总体评分**")
                            overall_col1, overall_col2, overall_col3 = st.columns(3)

                            with overall_col1:
                                st.metric("平均分", f"{summary['overall'].get('avg_score', 0):.3f}")
                            with overall_col2:
                                st.metric("最低分", f"{summary['overall'].get('min_score', 0):.3f}")
                            with overall_col3:
                                st.metric("最高分", f"{summary['overall'].get('max_score', 0):.3f}")

                        # Save results
                        results_dir = Path("data/eval_results")
                        results_dir.mkdir(parents=True, exist_ok=True)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_path = results_dir / f"eval_{timestamp}.json"

                        st.session_state.eval_runner.save_results(str(results_path))
                        st.info(f"📁 结果已保存: {results_path.name}")

                    except Exception as e:
                        st.error(f"评估失败: {str(e)}")

        except Exception as e:
            st.error(f"加载测试集失败: {str(e)}")

# Tab 2: Single Evaluation
with tab2:
    st.subheader("✍️ 单次评估")

    st.markdown("手动评估单个查询-答案对。")

    question = st.text_input("问题", placeholder="例如: 什么是机器学习？")
    answer = st.text_area("答案", placeholder="生成的答案...")

    contexts_input = st.text_area(
        "上下文（每行一个）",
        placeholder="检索到的上下文片段...\n每行一个上下文"
    )

    ground_truth = st.text_input("Ground Truth（可选）", placeholder="标准答案...")

    if st.button("评估", type="primary"):
        if question and answer and contexts_input:
            contexts = [c.strip() for c in contexts_input.split('\n') if c.strip()]

            with st.spinner("评估中..."):
                try:
                    result = st.session_state.eval_runner.composite_evaluator.evaluate_generation(
                        question=question,
                        answer=answer,
                        contexts=contexts,
                        ground_truth=ground_truth if ground_truth else None
                    )

                    st.success("✅ 评估完成")

                    if result.get("ragas"):
                        st.subheader("📊 Ragas 指标")

                        ragas_scores = result["ragas"]
                        cols = st.columns(min(len(ragas_scores), 4))

                        for i, (metric, score) in enumerate(ragas_scores.items()):
                            with cols[i % len(cols)]:
                                st.metric(metric.title(), f"{score:.3f}")

                    if not result.get("ragas_available"):
                        st.info("ℹ️ 使用降级模式评估（基于启发式规则）")

                except Exception as e:
                    st.error(f"评估失败: {str(e)}")
        else:
            st.warning("请填写问题、答案和上下文")

# Tab 3: Evaluation History
with tab3:
    st.subheader("📜 评估历史")

    results_dir = Path("data/eval_results")

    if results_dir.exists():
        result_files = sorted(results_dir.glob("eval_*.json"), reverse=True)

        if result_files:
            st.write(f"找到 {len(result_files)} 个评估结果")

            selected_result = st.selectbox(
                "选择评估结果",
                result_files,
                format_func=lambda x: x.name
            )

            if selected_result:
                with open(selected_result, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)

                st.markdown("**评估信息**")
                st.write(f"时间: {result_data.get('timestamp', 'N/A')}")
                st.write(f"测试用例数: {result_data.get('num_test_cases', 0)}")

                # Display summary
                summary = result_data.get("summary", {})

                if summary:
                    st.json(summary)

                # Download button
                st.download_button(
                    label="📥 下载完整结果",
                    data=json.dumps(result_data, indent=2, ensure_ascii=False),
                    file_name=selected_result.name,
                    mime="application/json"
                )
        else:
            st.info("暂无评估历史")
    else:
        st.info("暂无评估历史")

# Tab 4: Configuration
with tab4:
    st.subheader("⚙️ 评估配置")

    st.markdown("**Ragas 状态**")

    if st.session_state.ragas_available:
        st.success("✅ Ragas 已安装并可用")
    else:
        st.warning("⚠️ Ragas 未安装或配置不完整")
        st.markdown("""
        安装 Ragas:
        ```bash
        pip install ragas
        ```

        配置 LLM 和 Embedding（用于评估）:
        - 在 config/settings.yaml 中配置
        - 或设置环境变量
        """)

    st.markdown("---")

    st.markdown("**系统信息**")

    st.write(f"- Python 版本: {sys.version.split()[0]}")
    st.write(f"- 工作目录: {Path.cwd()}")
    st.write(f"- 测试集目录: data/test/")
    st.write(f"- 结果目录: data/eval_results/")
