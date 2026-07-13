"""Evaluation runner for scientific data catalogs-Vis."""
import yaml
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nl2vis_bench.models import DatasetSchema, QueryResult
from nl2vis_bench.translation import NLTranslator
from nl2vis_bench.execution import QueryExecutor
from nl2vis_bench.visualization import VizSelector
from nl2vis_bench.llm.base import LLMProvider


@dataclass
class QuestionResult:
    """Result of evaluating a single question."""
    question_id: str
    question: str
    category: str

    # Translation
    generated_query: str = ""
    translation_correct: bool = False
    columns_correct: bool = False
    translation_latency_ms: float = 0.0

    # Execution
    execution_success: bool = False
    execution_latency_ms: float = 0.0
    execution_error: str | None = None

    # Visualization
    selected_viz: str = ""
    viz_correct: bool = False

    # Overall
    error: str | None = None


@dataclass
class EvaluationResult:
    """Overall evaluation results."""
    total_questions: int = 0
    successful_translations: int = 0
    successful_executions: int = 0
    correct_visualizations: int = 0
    correct_columns: int = 0

    avg_translation_latency_ms: float = 0.0
    avg_execution_latency_ms: float = 0.0

    question_results: list[QuestionResult] = field(default_factory=list)

    @property
    def translation_accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.successful_translations / self.total_questions

    @property
    def execution_accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.successful_executions / self.total_questions

    @property
    def viz_accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.correct_visualizations / self.total_questions

    @property
    def column_accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.correct_columns / self.total_questions

    def summary(self) -> str:
        """Generate a summary report."""
        return f"""
Evaluation Results
==================
Total Questions: {self.total_questions}

Accuracy Metrics:
- Translation Accuracy: {self.translation_accuracy:.1%}
- Column Accuracy: {self.column_accuracy:.1%}
- Visualization Accuracy: {self.viz_accuracy:.1%}
- Execution Success Rate: {self.execution_accuracy:.1%}

Latency Metrics:
- Avg Translation Latency: {self.avg_translation_latency_ms:.0f}ms
- Avg Execution Latency: {self.avg_execution_latency_ms:.0f}ms

By Category:
"""


class EvaluationRunner:
    """Runs evaluation on test questions."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        questions_path: str | Path | None = None,
    ):
        self.llm = llm_provider
        self.translator = NLTranslator(llm_provider=llm_provider)
        self.executor = QueryExecutor()
        self.selector = VizSelector(llm_provider=llm_provider)

        if questions_path is None:
            questions_path = Path(__file__).parent.parent.parent / "data" / "evaluation" / "questions.yaml"

        self.questions = self._load_questions(questions_path)

    def _load_questions(self, path: Path | str) -> list[dict]:
        """Load questions from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("questions", [])

    def run(self, schema: DatasetSchema, df: "pd.DataFrame", max_questions: int | None = None) -> EvaluationResult:
        """Run evaluation on all questions."""
        import pandas as pd

        questions = self.questions
        if max_questions:
            questions = questions[:max_questions]

        result = EvaluationResult(total_questions=len(questions))
        translation_latencies = []
        execution_latencies = []

        for q in questions:
            qr = self._evaluate_question(q, schema, df)
            result.question_results.append(qr)

            if qr.translation_correct:
                result.successful_translations += 1
            if qr.columns_correct:
                result.correct_columns += 1
            if qr.execution_success:
                result.successful_executions += 1
            if qr.viz_correct:
                result.correct_visualizations += 1

            if qr.translation_latency_ms > 0:
                translation_latencies.append(qr.translation_latency_ms)
            if qr.execution_latency_ms > 0:
                execution_latencies.append(qr.execution_latency_ms)

        if translation_latencies:
            result.avg_translation_latency_ms = sum(translation_latencies) / len(translation_latencies)
        if execution_latencies:
            result.avg_execution_latency_ms = sum(execution_latencies) / len(execution_latencies)

        return result

    def _evaluate_question(self, q: dict, schema: DatasetSchema, df: "pd.DataFrame") -> QuestionResult:
        """Evaluate a single question."""
        qr = QuestionResult(
            question_id=q["id"],
            question=q["question"],
            category=q["category"],
        )

        try:
            # Step 1: Translate
            translation = self.translator.translate(q["question"], schema)
            qr.generated_query = translation.query
            qr.translation_latency_ms = translation.latency_ms

            # Check translation correctness
            expected_keywords = q.get("expected_query_contains", [])
            query_lower = translation.query.lower()
            qr.translation_correct = all(kw.lower() in query_lower for kw in expected_keywords)

            # Check columns
            expected_cols = set(q.get("expected_columns", []))
            actual_cols = set(translation.columns_used)
            qr.columns_correct = expected_cols == actual_cols or expected_cols.issubset(actual_cols)

            # Step 2: Execute
            exec_result = self.executor.execute(df, translation.query)
            qr.execution_success = exec_result.success
            qr.execution_latency_ms = exec_result.execution_time_ms
            if not exec_result.success:
                qr.execution_error = exec_result.error_message

            # Step 3: Select visualization
            if exec_result.success and exec_result.profile:
                viz_spec = self.selector.select(exec_result.profile, q["question"])
                qr.selected_viz = viz_spec.type
                qr.viz_correct = viz_spec.type == q.get("expected_viz", "")

        except Exception as e:
            qr.error = str(e)

        return qr
