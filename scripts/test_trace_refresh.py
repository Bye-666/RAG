"""End-to-end test for trace persistence across page refreshes.

This script simulates:
1. First session: Record traces
2. Page refresh (new session): Load traces from disk
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trace.trace_context import TraceContext, get_trace_recorder


def simulate_first_session():
    """Simulate first session where traces are recorded."""
    print("=== Simulating First Session ===")
    print("Recording traces...")

    # Record some ingestion traces
    with TraceContext("ingestion", "upload_doc1.pdf") as trace:
        trace.add_step("load", {"file": "doc1.pdf", "size": 1024})
        trace.add_step("split", {"chunks": 10})

    with TraceContext("ingestion", "upload_doc2.pdf") as trace:
        trace.add_step("load", {"file": "doc2.pdf", "size": 2048})
        trace.add_step("split", {"chunks": 15})

    # Check current session
    recorder = get_trace_recorder()
    traces = recorder.get_traces(trace_type="ingestion")
    print(f"Traces in current session: {len(traces)}")

    for i, trace in enumerate(traces, 1):
        print(f"  {i}. {trace['operation']}")

    return len(traces)


def simulate_page_refresh():
    """Simulate page refresh by creating new recorder from disk."""
    print("\n=== Simulating Page Refresh ===")
    print("Loading traces from disk...")

    # Force reload by clearing module cache and creating new recorder
    from src.trace.trace_context import TraceRecorder
    recorder = TraceRecorder()  # Uses default persist path

    traces = recorder.get_traces(trace_type="ingestion")
    print(f"Traces loaded after refresh: {len(traces)}")

    for i, trace in enumerate(traces, 1):
        print(f"  {i}. {trace['operation']}")

    return len(traces)


def main():
    """Run the end-to-end test."""
    print("Testing trace persistence across page refreshes...\n")

    # Clean up any existing trace file
    trace_file = Path("data/traces/traces.json")
    if trace_file.exists():
        trace_file.unlink()
        print(f"[Cleanup] Removed existing trace file\n")

    # First session
    count1 = simulate_first_session()

    # Page refresh
    count2 = simulate_page_refresh()

    # Verify
    print("\n=== Verification ===")
    if count2 >= count1:
        print(f"[SUCCESS] Traces persisted across refresh!")
        print(f"  Session 1: {count1} traces")
        print(f"  Session 2: {count2} traces")
        return True
    else:
        print(f"[FAILED] Traces not persisted!")
        print(f"  Session 1: {count1} traces")
        print(f"  Session 2: {count2} traces")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
