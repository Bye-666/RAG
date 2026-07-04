"""Trace context for tracking execution flow and timing."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
import time
import uuid


class TraceContext:
    """Context manager for tracing execution flow.

    Tracks:
    - Execution timing (start, end, duration)
    - Operation type (ingestion, query)
    - Steps and intermediate states
    - Nested sub-traces

    Usage:
        with TraceContext("ingestion", operation="pdf_ingestion") as trace:
            trace.add_step("load", {"file": "doc.pdf"})
            trace.add_step("split", {"chunks": 10})
            trace.finish({"status": "success"})
    """

    def __init__(
        self,
        trace_type: str,
        operation: str,
        trace_id: Optional[str] = None,
        parent_trace: Optional['TraceContext'] = None
    ):
        """Initialize TraceContext.

        Args:
            trace_type: Type of trace ("ingestion" or "query")
            operation: Operation name (e.g., "pdf_ingestion", "hybrid_search")
            trace_id: Optional trace ID (auto-generated if not provided)
            parent_trace: Optional parent trace for nested tracing
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        self.trace_type = trace_type
        self.operation = operation
        self.parent_trace = parent_trace

        # Timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.duration_ms: Optional[float] = None

        # Steps tracking
        self.steps: List[Dict[str, Any]] = []

        # Result
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

        # Sub-traces
        self.sub_traces: List['TraceContext'] = []

    def __enter__(self) -> 'TraceContext':
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is not None:
            self.error = f"{exc_type.__name__}: {exc_val}"

        self.finish()

        # Auto-record trace if it's a top-level trace (no parent)
        if self.parent_trace is None:
            get_trace_recorder().record(self)

        return False  # Don't suppress exceptions

    def start(self):
        """Start the trace."""
        self.start_time = datetime.now()

    def add_step(self, step_name: str, data: Optional[Dict[str, Any]] = None):
        """Add a step to the trace.

        Args:
            step_name: Name of the step
            data: Optional step data
        """
        step = {
            "name": step_name,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.steps.append(step)

    def finish(self, result: Optional[Dict[str, Any]] = None):
        """Finish the trace.

        Args:
            result: Optional result data
        """
        self.end_time = datetime.now()

        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        if result is not None:
            self.result = result

    def add_sub_trace(self, sub_trace: 'TraceContext'):
        """Add a nested sub-trace.

        Args:
            sub_trace: Child trace context
        """
        self.sub_traces.append(sub_trace)

    def to_dict(self) -> Dict[str, Any]:
        """Export trace as dictionary.

        Returns:
            Trace data dictionary
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "operation": self.operation,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "steps": self.steps,
            "result": self.result,
            "error": self.error,
            "sub_traces": [st.to_dict() for st in self.sub_traces],
            "parent_id": self.parent_trace.trace_id if self.parent_trace else None
        }

    @contextmanager
    def sub_trace(self, operation: str):
        """Create a nested sub-trace.

        Args:
            operation: Sub-operation name

        Yields:
            Sub-trace context
        """
        sub = TraceContext(
            trace_type=self.trace_type,
            operation=operation,
            parent_trace=self
        )

        with sub:
            yield sub

        self.add_sub_trace(sub)


class TraceRecorder:
    """Global trace recorder for collecting traces with file persistence."""

    def __init__(self, persist_path: Optional[str] = None):
        """Initialize TraceRecorder.

        Args:
            persist_path: Optional path to persist traces to disk
        """
        self.traces: List[TraceContext] = []
        self.persist_path = Path(persist_path) if persist_path else Path("data/traces/traces.json")

        # Load existing traces from disk
        self._load_from_disk()

    def _load_from_disk(self):
        """Load traces from disk if file exists."""
        if self.persist_path.exists():
            try:
                import json
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    traces_data = json.load(f)
                    # Convert dict back to TraceContext objects
                    for trace_dict in traces_data:
                        trace = TraceContext(
                            trace_dict['trace_type'],
                            trace_dict['operation'],
                            trace_id=trace_dict['trace_id']
                        )
                        # Restore trace state
                        if trace_dict.get('start_time'):
                            trace.start_time = datetime.fromisoformat(trace_dict['start_time'])
                        if trace_dict.get('end_time'):
                            trace.end_time = datetime.fromisoformat(trace_dict['end_time'])
                        trace.duration_ms = trace_dict.get('duration_ms')
                        trace.steps = trace_dict.get('steps', [])
                        trace.result = trace_dict.get('result')
                        trace.error = trace_dict.get('error')
                        self.traces.append(trace)
            except Exception:
                # If loading fails, start with empty traces
                pass

    def _save_to_disk(self):
        """Save traces to disk."""
        try:
            import json
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                traces_data = [t.to_dict() for t in self.traces]
                json.dump(traces_data, f, ensure_ascii=False, indent=2)
        except Exception:
            # Silently fail if save fails
            pass

    def record(self, trace: TraceContext):
        """Record a completed trace.

        Args:
            trace: Trace to record
        """
        self.traces.append(trace)
        self._save_to_disk()

    def get_traces(
        self,
        trace_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recorded traces.

        Args:
            trace_type: Optional filter by trace type
            limit: Optional limit number of results

        Returns:
            List of trace dictionaries
        """
        filtered = self.traces

        if trace_type:
            filtered = [t for t in filtered if t.trace_type == trace_type]

        if limit:
            filtered = filtered[-limit:]

        return [t.to_dict() for t in filtered]

    def clear(self):
        """Clear all recorded traces."""
        self.traces = []
        self._save_to_disk()


# Global trace recorder instance (for non-Streamlit contexts)
_global_recorder = None


def get_trace_recorder() -> TraceRecorder:
    """Get the trace recorder instance.

    In Streamlit context, returns the session-based recorder.
    Otherwise, returns a module-level global recorder.

    Returns:
        TraceRecorder instance
    """
    # Try to use Streamlit session state first
    try:
        import streamlit as st
        # Handle both real st.session_state and dict-based mocks
        session_state = st.session_state
        if isinstance(session_state, dict):
            # Mock session state (testing)
            if 'trace_recorder' not in session_state:
                session_state['trace_recorder'] = TraceRecorder()
            return session_state['trace_recorder']
        else:
            # Real Streamlit session state
            if 'trace_recorder' not in st.session_state:
                st.session_state.trace_recorder = TraceRecorder()
            return st.session_state.trace_recorder
    except (ImportError, RuntimeError, AttributeError):
        # Fallback to module-level global for non-Streamlit contexts
        global _global_recorder
        if _global_recorder is None:
            _global_recorder = TraceRecorder()
        return _global_recorder
