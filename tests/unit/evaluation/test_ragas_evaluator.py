"""Unit tests for RagasEvaluator."""

import pytest
from src.evaluation.ragas_evaluator import RagasEvaluator


def test_ragas_evaluator_init():
    """Test RagasEvaluator initialization."""
    evaluator = RagasEvaluator()
    assert evaluator is not None


def test_fallback_evaluate():
    """Test fallback evaluation when ragas is not available."""
    evaluator = RagasEvaluator()

    question = "What is machine learning?"
    answer = "Machine learning is a branch of artificial intelligence."
    contexts = [
        "Machine learning is a subset of AI.",
        "It involves training models on data."
    ]

    scores = evaluator.evaluate(question, answer, contexts)

    # Should return some scores
    assert isinstance(scores, dict)
    assert len(scores) > 0

    # Check score ranges
    for metric, score in scores.items():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_fallback_evaluate_with_ground_truth():
    """Test fallback evaluation with ground truth."""
    evaluator = RagasEvaluator()

    question = "What is Python?"
    answer = "Python is a programming language."
    contexts = ["Python is a high-level programming language."]
    ground_truth = "Python is a programming language used for software development."

    scores = evaluator.evaluate(question, answer, contexts, ground_truth)

    assert isinstance(scores, dict)
    assert "context_recall" in scores or len(scores) > 0


def test_fallback_evaluate_batch():
    """Test batch evaluation in fallback mode."""
    evaluator = RagasEvaluator()

    questions = [
        "What is AI?",
        "What is ML?"
    ]
    answers = [
        "AI is artificial intelligence.",
        "ML is machine learning."
    ]
    contexts_list = [
        ["AI stands for artificial intelligence."],
        ["ML is a subset of AI."]
    ]

    result = evaluator.evaluate_batch(questions, answers, contexts_list)

    assert isinstance(result, dict)
    assert "average_scores" in result
    assert "num_samples" in result
    assert result["num_samples"] == 2


def test_is_available():
    """Test checking if ragas is available."""
    evaluator = RagasEvaluator()
    available = evaluator.is_available()

    assert isinstance(available, bool)


def test_empty_contexts():
    """Test evaluation with empty contexts."""
    evaluator = RagasEvaluator()

    question = "What is test?"
    answer = "Test is a procedure."
    contexts = []

    scores = evaluator.evaluate(question, answer, contexts)

    # Should still return some scores
    assert isinstance(scores, dict)


def test_empty_answer():
    """Test evaluation with empty answer."""
    evaluator = RagasEvaluator()

    question = "What is empty?"
    answer = ""
    contexts = ["Some context here."]

    scores = evaluator.evaluate(question, answer, contexts)

    # Should handle gracefully
    assert isinstance(scores, dict)


def test_batch_with_ground_truths():
    """Test batch evaluation with ground truths."""
    evaluator = RagasEvaluator()

    questions = ["Q1?", "Q2?"]
    answers = ["A1", "A2"]
    contexts_list = [["C1"], ["C2"]]
    ground_truths = ["GT1", "GT2"]

    result = evaluator.evaluate_batch(
        questions, answers, contexts_list, ground_truths
    )

    assert isinstance(result, dict)
    assert result["num_samples"] == 2
