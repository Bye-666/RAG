"""Test trace persistence in Streamlit session state."""

import pytest
from unittest.mock import MagicMock, patch
from src.trace.trace_context import TraceContext, TraceRecorder, get_trace_recorder


def test_trace_recorder_fallback_to_module_global():
    """Test that trace recorder falls back to module-level global when Streamlit is not available."""
    # Mock streamlit to raise ImportError
    with patch.dict('sys.modules', {'streamlit': None}):
        recorder1 = get_trace_recorder()
        recorder2 = get_trace_recorder()

        # Should return the same instance
        assert recorder1 is recorder2

        # Record a trace
        with TraceContext('ingestion', 'test_op') as trace:
            trace.add_step('step1', {})

        # Should be able to retrieve it
        traces = recorder1.get_traces(trace_type='ingestion')
        assert len(traces) == 1


def test_trace_recorder_uses_session_state():
    """Test that trace recorder uses Streamlit session state when available."""
    # Mock streamlit session state
    mock_st = MagicMock()
    mock_session_state = {}
    mock_st.session_state = mock_session_state

    with patch.dict('sys.modules', {'streamlit': mock_st}):
        # First call should create a new recorder in session state
        recorder1 = get_trace_recorder()
        assert 'trace_recorder' in mock_session_state

        # Second call should return the same recorder
        recorder2 = get_trace_recorder()
        assert recorder1 is recorder2

        # Record a trace
        with TraceContext('query', 'test_search') as trace:
            trace.add_step('search', {'query': 'test'})

        # Should be able to retrieve it from the session-based recorder
        traces = recorder1.get_traces(trace_type='query')
        assert len(traces) == 1
        assert traces[0]['operation'] == 'test_search'


def test_trace_context_auto_records():
    """Test that TraceContext automatically records to the trace recorder."""
    # Clear any existing traces
    recorder = get_trace_recorder()
    recorder.clear()

    # Create and finish a trace
    with TraceContext('ingestion', 'auto_record_test') as trace:
        trace.add_step('step1', {'data': 'value1'})
        trace.add_step('step2', {'data': 'value2'})

    # Should be automatically recorded
    traces = recorder.get_traces(trace_type='ingestion')
    assert len(traces) == 1
    assert traces[0]['operation'] == 'auto_record_test'
    assert len(traces[0]['steps']) == 2


def test_trace_filtering_by_type():
    """Test filtering traces by type."""
    recorder = get_trace_recorder()
    recorder.clear()

    # Record multiple trace types
    with TraceContext('ingestion', 'ingest_op') as trace:
        trace.add_step('load', {})

    with TraceContext('query', 'search_op') as trace:
        trace.add_step('search', {})

    with TraceContext('ingestion', 'ingest_op2') as trace:
        trace.add_step('load', {})

    # Filter by type
    ingestion_traces = recorder.get_traces(trace_type='ingestion')
    query_traces = recorder.get_traces(trace_type='query')
    all_traces = recorder.get_traces()

    assert len(ingestion_traces) == 2
    assert len(query_traces) == 1
    assert len(all_traces) == 3


def test_trace_limit():
    """Test limiting number of returned traces."""
    recorder = get_trace_recorder()
    recorder.clear()

    # Record 5 traces
    for i in range(5):
        with TraceContext('ingestion', f'op_{i}') as trace:
            trace.add_step('step', {})

    # Get last 3 traces
    traces = recorder.get_traces(limit=3)
    assert len(traces) == 3

    # Should return the most recent ones
    assert traces[0]['operation'] == 'op_2'
    assert traces[1]['operation'] == 'op_3'
    assert traces[2]['operation'] == 'op_4'
