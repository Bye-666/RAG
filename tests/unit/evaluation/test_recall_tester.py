"""Unit tests for RecallRegressionTester."""

import pytest
import json
import tempfile
from pathlib import Path
from src.evaluation.recall_tester import RecallRegressionTester


@pytest.fixture
def sample_baseline():
    """Create sample baseline data."""
    return {
        "created_at": "2026-01-01T00:00:00",
        "num_cases": 3,
        "avg_recall": 0.8,
        "test_cases": [
            {"id": "test_001", "question": "Q1", "recall": 0.8, "retrieved_count": 4, "relevant_count": 5},
            {"id": "test_002", "question": "Q2", "recall": 0.9, "retrieved_count": 9, "relevant_count": 10},
            {"id": "test_003", "question": "Q3", "recall": 0.7, "retrieved_count": 7, "relevant_count": 10}
        ]
    }


@pytest.fixture
def temp_baseline_file(sample_baseline):
    """Create temporary baseline file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(sample_baseline, f)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


def test_recall_tester_init():
    """Test RecallRegressionTester initialization."""
    tester = RecallRegressionTester()
    assert tester is not None
    assert tester.threshold == 0.05


def test_load_baseline(temp_baseline_file):
    """Test loading baseline."""
    tester = RecallRegressionTester()
    tester.load_baseline(temp_baseline_file)

    assert tester.baseline_data is not None
    assert tester.baseline_data["num_cases"] == 3


def test_load_nonexistent_baseline():
    """Test loading non-existent baseline."""
    tester = RecallRegressionTester()

    with pytest.raises(FileNotFoundError):
        tester.load_baseline("nonexistent.json")


def test_create_baseline():
    """Test creating baseline."""
    test_cases = [
        {
            "id": "test_001",
            "question": "Q1",
            "retrieved_docs": ["doc1", "doc2"],
            "relevant_docs": ["doc1", "doc2", "doc3"]
        },
        {
            "id": "test_002",
            "question": "Q2",
            "retrieved_docs": ["doc1"],
            "relevant_docs": ["doc1"]
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "baseline.json"

        tester = RecallRegressionTester()
        baseline = tester.create_baseline(test_cases, str(output_path))

        assert baseline["num_cases"] == 2
        assert "avg_recall" in baseline
        assert output_path.exists()


def test_test_regression_no_regression(temp_baseline_file):
    """Test regression testing with no regression."""
    tester = RecallRegressionTester(baseline_path=temp_baseline_file)

    # Same performance as baseline
    test_cases = [
        {"id": "test_001", "retrieved_docs": ["d1", "d2", "d3", "d4"], "relevant_docs": ["d1", "d2", "d3", "d4", "d5"]},
        {"id": "test_002", "retrieved_docs": list(range(9)), "relevant_docs": list(range(10))},
        {"id": "test_003", "retrieved_docs": list(range(7)), "relevant_docs": list(range(10))}
    ]

    result = tester.test_regression(test_cases)

    assert result["has_regression"] is False
    assert result["num_regressions"] == 0


def test_test_regression_with_regression(temp_baseline_file):
    """Test regression testing with regression detected."""
    tester = RecallRegressionTester(baseline_path=temp_baseline_file, threshold=0.05)

    # Lower performance than baseline
    test_cases = [
        {"id": "test_001", "retrieved_docs": ["d1"], "relevant_docs": ["d1", "d2", "d3", "d4", "d5"]},  # 0.2 recall (was 0.8)
        {"id": "test_002", "retrieved_docs": list(range(9)), "relevant_docs": list(range(10))},  # 0.9 (same)
        {"id": "test_003", "retrieved_docs": list(range(7)), "relevant_docs": list(range(10))}   # 0.7 (same)
    ]

    result = tester.test_regression(test_cases)

    assert result["has_regression"] is True
    assert result["num_regressions"] >= 1


def test_test_regression_without_baseline():
    """Test regression testing without baseline."""
    tester = RecallRegressionTester()

    with pytest.raises(ValueError):
        tester.test_regression([])


def test_save_results(temp_baseline_file):
    """Test saving regression test results."""
    tester = RecallRegressionTester(baseline_path=temp_baseline_file)

    test_cases = [
        {"id": "test_001", "retrieved_docs": ["d1"], "relevant_docs": ["d1", "d2"]},
    ]

    tester.test_regression(test_cases)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.json"

        tester.save_results(str(output_path))

        assert output_path.exists()

        with open(output_path, 'r', encoding='utf-8') as f:
            saved_results = json.load(f)

        assert "results" in saved_results
        assert "threshold" in saved_results


def test_save_results_without_running():
    """Test saving results without running test."""
    tester = RecallRegressionTester()

    with pytest.raises(ValueError):
        tester.save_results("output.json")


def test_get_regression_cases(temp_baseline_file):
    """Test getting regression cases."""
    tester = RecallRegressionTester(baseline_path=temp_baseline_file, threshold=0.05)

    test_cases = [
        {"id": "test_001", "retrieved_docs": ["d1"], "relevant_docs": ["d1", "d2", "d3", "d4", "d5"]},
        {"id": "test_002", "retrieved_docs": list(range(9)), "relevant_docs": list(range(10))},
    ]

    tester.test_regression(test_cases)

    regression_cases = tester.get_regression_cases()

    assert isinstance(regression_cases, list)


def test_update_threshold():
    """Test updating threshold."""
    tester = RecallRegressionTester(threshold=0.05)

    tester.update_threshold(0.1)
    assert tester.threshold == 0.1


def test_update_threshold_invalid():
    """Test updating threshold with invalid value."""
    tester = RecallRegressionTester()

    with pytest.raises(ValueError):
        tester.update_threshold(1.5)


def test_new_test_case_not_in_baseline(temp_baseline_file):
    """Test handling new test case not in baseline."""
    tester = RecallRegressionTester(baseline_path=temp_baseline_file)

    test_cases = [
        {"id": "test_001", "retrieved_docs": ["d1"], "relevant_docs": ["d1", "d2"]},
        {"id": "test_999", "retrieved_docs": ["d1"], "relevant_docs": ["d1", "d2"]},  # New case
    ]

    result = tester.test_regression(test_cases)

    assert result["num_cases"] == 2
