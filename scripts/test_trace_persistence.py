"""Test script to verify trace persistence in Streamlit context.

Usage:
    python scripts/test_trace_persistence.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trace.trace_context import TraceContext, get_trace_recorder


def test_trace_persistence():
    """Test that traces persist across get_trace_recorder() calls."""
    print("Testing trace persistence...")

    # Simulate first operation (e.g., ingestion)
    print("\n1. Creating first trace (ingestion)...")
    with TraceContext("ingestion", "test_ingestion_1") as trace:
        trace.add_step("load", {"file": "test.pdf"})
        trace.add_step("split", {"chunks": 10})

    # Get recorder and check
    recorder1 = get_trace_recorder()
    traces1 = recorder1.get_traces(trace_type="ingestion")
    print(f"   [OK] Traces after first operation: {len(traces1)}")

    # Simulate second operation
    print("\n2. Creating second trace (ingestion)...")
    with TraceContext("ingestion", "test_ingestion_2") as trace:
        trace.add_step("load", {"file": "test2.pdf"})
        trace.add_step("split", {"chunks": 15})

    # Get recorder again (simulates page refresh)
    recorder2 = get_trace_recorder()
    traces2 = recorder2.get_traces(trace_type="ingestion")
    print(f"   [OK] Traces after second operation: {len(traces2)}")

    # Verify persistence
    print("\n3. Verifying persistence...")
    assert recorder1 is recorder2, "Recorders should be the same instance"
    assert len(traces2) == 2, f"Expected 2 traces, got {len(traces2)}"
    print("   [OK] Same recorder instance")
    print("   [OK] All traces persisted")

    # Verify trace details
    print("\n4. Verifying trace details...")
    for i, trace in enumerate(traces2, 1):
        print(f"   Trace {i}: {trace['operation']}")
        print(f"     - Steps: {len(trace['steps'])}")
        print(f"     - Duration: {trace.get('duration_ms', 0):.2f} ms")

    print("\n[SUCCESS] All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_trace_persistence()
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
