"""Clean up script to stop all Python processes and prepare for Streamlit restart.

Usage:
    python scripts/cleanup_processes.py
"""

import subprocess
import sys
import time
from pathlib import Path


def stop_all_python_processes():
    """Stop all Python processes to release database locks."""
    print("Stopping all Python processes...")

    try:
        if sys.platform == "win32":
            # Windows
            subprocess.run(
                ["taskkill", "/F", "/IM", "python.exe"],
                capture_output=True,
                text=True
            )
            subprocess.run(
                ["taskkill", "/F", "/IM", "pythonw.exe"],
                capture_output=True,
                text=True
            )
        else:
            # Linux/Mac
            subprocess.run(["pkill", "-9", "python"], capture_output=True)

        print("[OK] Python processes stopped")

        # Wait for processes to fully terminate
        time.sleep(2)

    except Exception as e:
        print(f"[WARNING] Could not stop processes: {e}")


def check_trace_file():
    """Check if trace file exists and is valid."""
    trace_file = Path("data/traces/traces.json")

    if trace_file.exists():
        size = trace_file.stat().st_size
        print(f"[OK] Trace file exists: {trace_file} ({size} bytes)")
        return True
    else:
        print(f"[INFO] Trace file does not exist yet: {trace_file}")
        return False


def main():
    """Run cleanup."""
    print("=== Cleanup Script ===\n")

    # Stop processes
    stop_all_python_processes()

    # Check trace file
    print("\nChecking trace file...")
    check_trace_file()

    print("\n=== Cleanup Complete ===")
    print("\nYou can now start Streamlit:")
    print("  streamlit run src/dashboard/streamlit_app.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
