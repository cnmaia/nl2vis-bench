"""Experiment configuration and runner."""

import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from nl2vis_bench.models import DatasetSchema
from nl2vis_bench.translation import NLTranslator
from nl2vis_bench.execution import QueryExecutor
from nl2vis_bench.visualization import VizSelector
from nl2vis_bench.llm.base import LLMProvider

from .ground_truth import compare_queries
from .detailed_logger import DetailedLogger, QuestionLog


@dataclass
class ExperimentConfig:
    """Configuration for a validation experiment."""

    model: str
    enrichment: bool
    questions_path: str | Path
    output_path: str | Path
    dataset_path: str | Path | None = None
    schema_path: str | Path | None = None

    @property
    def provider(self) -> str:
        """Derive provider from model name."""
        if self.model.startswith("gemini"):
            return "gemini"
        elif self.model.startswith("gpt"):
            return "openai"
        else:
            raise ValueError(f"Unknown model provider for: {self.model}")


@dataclass
class QuestionResult:
    """Result of evaluating a single question."""

    question_id: str
    question: str
    category: str

    # Query
    generated_query: str = ""
    expected_query: str = ""
    ground_truth_match: bool = False
    columns_correct: bool = False
    execution_success: bool = False

    # Visualization
    selected_viz: str = ""
    expected_viz: str = ""
    viz_type_correct: bool = False

    selected_x: str = ""
    selected_y: str | None = None
    expected_x: str = ""
    expected_y: str | None = None
    axes_correct: bool = False
    viz_correct: bool = False

    # Meta
    needs_manual_review: bool = False
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class ExperimentResult:
    """Overall experiment results."""

    metadata: dict = field(default_factory=dict)
    results: list[QuestionResult] = field(default_factory=list)

    def _compute_accuracy(self, predicate: Callable[[QuestionResult], bool]) -> float:
        """Compute accuracy rate for questions matching predicate."""
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if predicate(r))
        return correct / len(self.results)

    @property
    def ground_truth_match_rate(self) -> float:
        return self._compute_accuracy(lambda r: r.ground_truth_match)

    @property
    def execution_success_rate(self) -> float:
        return self._compute_accuracy(lambda r: r.execution_success)

    @property
    def viz_type_accuracy(self) -> float:
        return self._compute_accuracy(lambda r: r.viz_type_correct)

    @property
    def axes_accuracy(self) -> float:
        return self._compute_accuracy(lambda r: r.axes_correct)

    @property
    def viz_accuracy(self) -> float:
        return self._compute_accuracy(lambda r: r.viz_correct)

    @property
    def columns_accuracy(self) -> float:
        return self._compute_accuracy(lambda r: r.columns_correct)

    @property
    def manual_review_ids(self) -> list[str]:
        return [r.question_id for r in self.results if r.needs_manual_review]


class ExperimentRunner:
    """Runs validation experiments."""

    def __init__(self, llm_provider: LLMProvider, detailed_logging: bool = False):
        self.llm = llm_provider
        self.translator = NLTranslator(llm_provider=llm_provider)
        self.executor = QueryExecutor()
        self.selector = VizSelector(llm_provider=llm_provider)
        self.logger = DetailedLogger() if detailed_logging else None

    def run_single_question(
        self,
        question: dict,
        schema: DatasetSchema,
        df: "pd.DataFrame",
        use_enrichment: bool = True,
    ) -> QuestionResult:
        """Evaluate a single question against ground truth."""
        qr = QuestionResult(
            question_id=question["id"],
            question=question["question"],
            category=question["category"],
            expected_query=question.get("expected_query", ""),
            expected_viz=question.get("expected_viz", ""),
            expected_x=question.get("expected_x", ""),
            expected_y=question.get("expected_y"),
        )

        start_time = time.time()

        try:
            # Step 1: Translate
            translation = self.translator.translate(
                question["question"],
                schema,
                use_enrichment=use_enrichment
            )
            qr.generated_query = translation.query

            # Compare with ground truth
            match_result = compare_queries(translation.query, qr.expected_query)
            qr.ground_truth_match = match_result.matches

            # Check columns
            expected_cols = set(question.get("expected_columns", []))
            actual_cols = set(translation.columns_used)
            qr.columns_correct = expected_cols.issubset(actual_cols)

            # Step 2: Execute
            exec_result = self.executor.execute(df, translation.query)
            qr.execution_success = exec_result.success

            if not qr.ground_truth_match and qr.execution_success:
                qr.needs_manual_review = True

            # Step 3: Select visualization
            if exec_result.success and exec_result.profile:
                viz_spec = self.selector.select(exec_result.profile, question["question"])
                qr.selected_viz = viz_spec.type
                qr.selected_x = viz_spec.x
                qr.selected_y = viz_spec.y

                # Check visualization correctness
                qr.viz_type_correct = viz_spec.type == qr.expected_viz

                # Check axes
                if qr.expected_y is None:
                    # Histogram: only check x
                    qr.axes_correct = qr.selected_x == qr.expected_x
                else:
                    # Line/scatter: check both
                    qr.axes_correct = (
                        qr.selected_x == qr.expected_x and
                        qr.selected_y == qr.expected_y
                    )

                qr.viz_correct = qr.viz_type_correct and qr.axes_correct

        except Exception as e:
            qr.error = str(e)

        qr.latency_ms = (time.time() - start_time) * 1000

        # Log detailed information if enabled
        if self.logger:
            # Build expected and actual columns lists
            expected_columns = [qr.expected_x]
            if qr.expected_y is not None:
                expected_columns.append(qr.expected_y)

            actual_columns = [qr.selected_x] if qr.selected_x else []
            if qr.selected_y is not None:
                actual_columns.append(qr.selected_y)

            log_entry = QuestionLog(
                question_id=qr.question_id,
                question=qr.question,
                generated_code=qr.generated_query,
                execution_success=qr.execution_success,
                execution_error=qr.error,
                expected_viz_type=qr.expected_viz,
                actual_viz_type=qr.selected_viz if qr.selected_viz else None,
                expected_columns=expected_columns,
                actual_columns=actual_columns,
                viz_type_correct=qr.viz_type_correct,
                axes_correct=qr.axes_correct,
                overall_correct=qr.viz_correct
            )
            self.logger.log(log_entry)

        return qr

    def run(
        self,
        config: ExperimentConfig,
        schema: DatasetSchema,
        df: "pd.DataFrame",
        max_questions: int | None = None,
    ) -> ExperimentResult:
        """Run full experiment on all questions."""
        from datetime import datetime

        # Load questions
        with open(config.questions_path) as f:
            data = yaml.safe_load(f)

        questions = data.get("questions", [])
        if max_questions:
            questions = questions[:max_questions]

        result = ExperimentResult(
            metadata={
                "model": self.llm.model_name(),
                "enrichment": config.enrichment,
                "dataset": data.get("dataset", "unknown"),
                "executed_at": datetime.now().isoformat(),
                "total_questions": len(questions),
            }
        )

        for q in questions:
            qr = self.run_single_question(
                question=q,
                schema=schema,
                df=df,
                use_enrichment=config.enrichment,
            )
            result.results.append(qr)

        return result
