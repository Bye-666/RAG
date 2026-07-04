"""Shared session state initialization for all dashboard pages."""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def init_session_state():
    """Initialize session state variables if not already initialized."""

    # Config
    if 'config' not in st.session_state:
        from src.config.settings import Settings
        try:
            st.session_state.config = Settings.from_yaml("config/settings.yaml")
        except Exception as e:
            st.session_state.config = Settings()

    # Component Loader
    if 'component_loader' not in st.session_state:
        from src.libs.loader import ComponentLoader
        try:
            st.session_state.component_loader = ComponentLoader(st.session_state.config)
        except Exception as e:
            st.warning(f"组件加载器初始化失败: {str(e)}")
            st.session_state.component_loader = None

    # Trace Recorder - now handled automatically by get_trace_recorder()
    # The recorder will be stored in session_state when first accessed
    if 'trace_recorder' not in st.session_state:
        from src.trace.trace_context import get_trace_recorder
        # Initialize by calling get_trace_recorder() which handles session_state
        get_trace_recorder()
