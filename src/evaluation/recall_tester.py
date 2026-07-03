"""Recall regression testing for RAG system.

Monitors retrieval quality over time to detect regressions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class RecallRegressionTester:
    """Test for recall regressions in retrieval system.

    Maintains a baseline and detects when recall drops below threshold.
    """

    def __init__(
        self,
        baseline_path: Optional[str] = None,
        threshold: float = 0.05  # 5% drop tolerance
    ):
        """Initialize RecallRegressionTester.

        Args:
            baseline_path: Path to baseline recall data
            threshold: Maximum acceptable recall drop (0.05 = 5%)
        """
        self.baseline_path = baseline_path
        self.threshold = threshold
        self.baseline_data: Optional[Dict[str, Any]] = None
        self.current_results: List[Dict[str, Any]] = []

        if baseline_path:
            self.load_baseline(baseline_path)

    def load_baseline(self, baseline_path: str) -> None:
        """Load baseline recall data.

        Args:
            baseline_path: Path to baseline JSON file
        """
        path_obj = Path(baseline_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

        with open(path_obj, 'r', encoding='utf-8') as f:
            self.baseline_data = json.load(f)

        self.baseline_path = baseline_path

    def create_baseline(
        self,
        test_cases: List[Dict[str, Any]],
        output_path: str
    ) -> Dict[str, Any]:
        """Create a new baseline from test cases.

        Args:
            test_cases: List of test cases with retrieval results
            output_path: Where to save the baseline

        Returns:
            Baseline data
        """
        baseline = {
            "created_at": datetime.now().isoformat(),
            "num_cases": len(test_cases),
            "test_cases": []
        }

        total_recall = 0.0

        for case in test_cases:
            retrieved = set(case.get("retrieved_docs", []))
            relevant = set(case.get("relevant_docs", []))

            if relevant:
                recall = len(retrieved & relevant) / len(relevant)
            else:
                recall = 0.0

            total_recall += recall

            baseline["test_cases"].append({
                "id": case.get("id", "unknown"),
                "question": case.get("question", ""),
                "recall": recall,
                "retrieved_count": len(retrieved),
                "relevant_count": len(relevant)
            })

        baseline["avg_recall"] = total_recall / len(test_cases) if test_cases else 0.0

        # Save baseline
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)

        self.baseline_data = baseline
        self.baseline_path = output_path

        return baseline

    def test_regression(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test for recall regression against baseline.

        Args:
            test_cases: Current test cases with retrieval results

        Returns:
            Regression test results
        """
        if not self.baseline_data:
            raise ValueError("No baseline loaded. Call load_baseline() or create_baseline() first.")

        self.current_results = []
        regressions = []

        # Create case ID to baseline mapping
        baseline_map = {
            case["id"]: case
            for case in self.baseline_data["test_cases"]
        }

        total_recall = 0.0

        for case in test_cases:
            case_id = case.get("id", "unknown")

            # Calculate current recall
            retrieved = set(case.get("retrieved_docs", []))
            relevant = set(case.get("relevant_docs", []))

            if relevant:
                current_recall = len(retrieved & relevant) / len(relevant)
            else:
                current_recall = 0.0

            total_recall += current_recall

            # Compare with baseline
            baseline_case = baseline_map.get(case_id)

            if baseline_case:
                baseline_recall = baseline_case["recall"]
                recall_drop = baseline_recall - current_recall

                # Check for regression
                if recall_drop > self.threshold:
                    regressions.append({
                        "id": case_id,
                        "question": case.get("question", ""),
                        "baseline_recall": baseline_recall,
                        "current_recall": current_recall,
                        "drop": recall_drop,
                        "drop_percentage": recall_drop / baseline_recall * 100 if baseline_recall > 0 else 0
                    })

                self.current_results.append({
                    "id": case_id,
                    "baseline_recall": baseline_recall,
                    "current_recall": current_recall,
                    "drop": recall_drop,
                    "is_regression": recall_drop > self.threshold
                })
            else:
                # New test case not in baseline
                self.current_results.append({
                    "id": case_id,
                    "baseline_recall": None,
                    "current_recall": current_recall,
                    "drop": None,
                    "is_regression": False
                })

        # Compute averages
        avg_current_recall = total_recall / len(test_cases) if test_cases else 0.0
        avg_baseline_recall = self.baseline_data.get("avg_recall", 0.0)
        avg_drop = avg_baseline_recall - avg_current_recall

        result = {
            "timestamp": datetime.now().isoformat(),
            "baseline_path": self.baseline_path,
            "num_cases": len(test_cases),
            "num_regressions": len(regressions),
            "threshold": self.threshold,
            "avg_baseline_recall": avg_baseline_recall,
            "avg_current_recall": avg_current_recall,
            "avg_drop": avg_drop,
            "has_regression": len(regressions) > 0 or avg_drop > self.threshold,
            "regressions": regressions,
            "summary": self._generate_summary(regressions, avg_drop)
        }

        return result

    def _generate_summary(
        self,
        regressions: List[Dict[str, Any]],
        avg_drop: float
    ) -> str:
        """Generate human-readable summary.

        Args:
            regressions: List of regression cases
            avg_drop: Average recall drop

        Returns:
            Summary text
        """
        if not regressions and avg_drop <= self.threshold:
            return "✅ No regressions detected. Recall is stable."

        summary_parts = []

        if regressions:
            summary_parts.append(f"⚠️ {len(regressions)} test case(s) regressed:")
            for reg in regressions[:3]:  # Show top 3
                summary_parts.append(
                    f"  - {reg['id']}: {reg['baseline_recall']:.3f} → {reg['current_recall']:.3f} "
                    f"(-{reg['drop_percentage']:.1f}%)"
                )
            if len(regressions) > 3:
                summary_parts.append(f"  ... and {len(regressions) - 3} more")

        if avg_drop > self.threshold:
            summary_parts.append(
                f"⚠️ Average recall dropped by {avg_drop:.3f} ({avg_drop/self.baseline_data.get('avg_recall', 1)*100:.1f}%)"
            )

        return "\n".join(summary_parts)

    def save_results(self, output_path: str) -> None:
        """Save regression test results.

        Args:
            output_path: Where to save results
        """
        if not self.current_results:
            raise ValueError("No results to save. Run test_regression() first.")

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "baseline_path": self.baseline_path,
            "threshold": self.threshold,
            "results": self.current_results
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def get_regression_cases(self) -> List[Dict[str, Any]]:
        """Get list of regressed test cases.

        Returns:
            List of cases with regression
        """
        return [
            case for case in self.current_results
            if case.get("is_regression", False)
        ]

    def update_threshold(self, new_threshold: float) -> None:
        """Update regression threshold.

        Args:
            new_threshold: New threshold value (e.g., 0.05 for 5%)
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        self.threshold = new_threshold
