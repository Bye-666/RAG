"""Ragas-based RAG evaluation system.

Integrates the ragas library for professional RAG quality metrics.
"""

from typing import List, Dict, Any, Optional
import warnings


class RagasEvaluator:
    """RAG evaluation using ragas framework.

    Provides industry-standard metrics:
    - Faithfulness: How grounded the answer is in the context
    - Answer Relevancy: How relevant the answer is to the question
    - Context Precision: How precise the retrieved context is
    - Context Recall: How much relevant context was retrieved
    """

    def __init__(self, llm: Optional[Any] = None, embedding: Optional[Any] = None):
        """Initialize RagasEvaluator.

        Args:
            llm: Language model for evaluation (optional)
            embedding: Embedding model for evaluation (optional)
        """
        self.llm = llm
        self.embedding = embedding
        self._ragas_available = False

        # Try to import ragas (use new API to avoid deprecation warnings)
        try:
            import ragas
            try:
                # Try new API first (v1.0+)
                from ragas.metrics.collections import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                )
            except ImportError:
                # Fall back to old API
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                )
            from ragas import evaluate as ragas_evaluate
            from datasets import Dataset

            self._ragas_available = True
            self._ragas = ragas
            self._faithfulness = faithfulness
            self._answer_relevancy = answer_relevancy
            self._context_precision = context_precision
            self._context_recall = context_recall
            self._ragas_evaluate = ragas_evaluate
            self._Dataset = Dataset

        except ImportError:
            warnings.warn(
                "ragas library not installed. Install with: pip install ragas\n"
                "RagasEvaluator will operate in fallback mode with basic metrics."
            )

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate a single RAG query-answer pair.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
            ground_truth: Ground truth answer (optional, for context_recall)

        Returns:
            Dictionary of metric scores
        """
        if not self._ragas_available:
            return self._fallback_evaluate(question, answer, contexts, ground_truth)

        try:
            # Prepare data in ragas format
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }

            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = self._Dataset.from_dict(data)

            # Select metrics based on available data
            metrics = [
                self._faithfulness,
                self._answer_relevancy,
            ]

            if ground_truth:
                metrics.append(self._context_recall)
                metrics.append(self._context_precision)

            # Evaluate
            result = self._ragas_evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embedding
            )

            # Extract scores
            scores = {}
            for metric_name, score in result.items():
                if isinstance(score, (int, float)):
                    scores[metric_name] = float(score)

            return scores

        except Exception as e:
            warnings.warn(f"Ragas evaluation failed: {e}. Using fallback.")
            return self._fallback_evaluate(question, answer, contexts, ground_truth)

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate multiple RAG query-answer pairs.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists
            ground_truths: List of ground truth answers (optional)

        Returns:
            Dictionary with average scores and individual results
        """
        if not self._ragas_available:
            return self._fallback_evaluate_batch(
                questions, answers, contexts_list, ground_truths
            )

        try:
            # Prepare data
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
            }

            if ground_truths:
                data["ground_truth"] = ground_truths

            dataset = self._Dataset.from_dict(data)

            # Select metrics
            metrics = [
                self._faithfulness,
                self._answer_relevancy,
            ]

            if ground_truths:
                metrics.append(self._context_recall)
                metrics.append(self._context_precision)

            # Evaluate
            result = self._ragas_evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embedding
            )

            return {
                "average_scores": dict(result),
                "num_samples": len(questions)
            }

        except Exception as e:
            warnings.warn(f"Batch evaluation failed: {e}. Using fallback.")
            return self._fallback_evaluate_batch(
                questions, answers, contexts_list, ground_truths
            )

    def _fallback_evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Fallback evaluation using basic heuristics.

        Used when ragas is not available.
        """
        scores = {}

        # Basic faithfulness: check if answer tokens appear in contexts
        if contexts:
            context_text = " ".join(contexts).lower()
            answer_tokens = set(answer.lower().split())
            context_tokens = set(context_text.split())

            if answer_tokens:
                overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
                scores["faithfulness"] = overlap

        # Basic answer relevancy: check question-answer overlap
        question_tokens = set(question.lower().split())
        answer_tokens = set(answer.lower().split())

        if question_tokens:
            overlap = len(question_tokens & answer_tokens) / len(question_tokens)
            scores["answer_relevancy"] = overlap

        # Context recall: if ground truth available
        if ground_truth and contexts:
            context_text = " ".join(contexts).lower()
            gt_tokens = set(ground_truth.lower().split())
            context_tokens = set(context_text.split())

            if gt_tokens:
                overlap = len(gt_tokens & context_tokens) / len(gt_tokens)
                scores["context_recall"] = overlap

        return scores

    def _fallback_evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Fallback batch evaluation."""
        all_scores = []

        for i in range(len(questions)):
            gt = ground_truths[i] if ground_truths else None
            scores = self._fallback_evaluate(
                questions[i],
                answers[i],
                contexts_list[i],
                gt
            )
            all_scores.append(scores)

        # Calculate averages
        if all_scores:
            metric_names = all_scores[0].keys()
            avg_scores = {}
            for metric in metric_names:
                values = [s[metric] for s in all_scores if metric in s]
                if values:
                    avg_scores[metric] = sum(values) / len(values)

            return {
                "average_scores": avg_scores,
                "num_samples": len(questions)
            }

        return {"average_scores": {}, "num_samples": 0}

    def is_available(self) -> bool:
        """Check if ragas library is available.

        Returns:
            True if ragas is installed and can be used
        """
        return self._ragas_available
