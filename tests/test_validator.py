"""Tests for validator module."""

import pytest
from nl2vis_bench.validator.ground_truth import normalize_query, compare_queries
from nl2vis_bench.validator.experiment import ExperimentConfig


class TestNormalizeQuery:
    """Tests for query normalization."""

    def test_removes_whitespace(self):
        query = "df[ 'col1' ]"
        assert normalize_query(query) == "df['col1']"

    def test_standardizes_quotes(self):
        query = 'df["col1"]'
        assert normalize_query(query) == "df['col1']"

    def test_lowercases(self):
        query = "DF['COL1']"
        assert normalize_query(query) == "df['col1']"

    def test_sorts_column_lists(self):
        query = "df[['b', 'a']]"
        assert normalize_query(query) == "df[['a','b']]"

    def test_complex_query(self):
        q1 = 'df[ [ "time" , "NEE_CUT_REF" ] ]'
        q2 = "df[['NEE_CUT_REF', 'time']]"
        # Both should normalize to same thing
        assert normalize_query(q1) == normalize_query(q2)


class TestCompareQueries:
    """Tests for query comparison."""

    def test_identical_queries_match(self):
        result = compare_queries(
            "df[['time', 'NEE_CUT_REF']]", "df[['time', 'NEE_CUT_REF']]"
        )
        assert result.matches is True

    def test_different_whitespace_matches(self):
        result = compare_queries(
            "df[ [ 'time' , 'NEE_CUT_REF' ] ]", "df[['time', 'NEE_CUT_REF']]"
        )
        assert result.matches is True

    def test_different_column_order_matches(self):
        result = compare_queries(
            "df[['NEE_CUT_REF', 'time']]", "df[['time', 'NEE_CUT_REF']]"
        )
        assert result.matches is True

    def test_different_queries_no_match(self):
        result = compare_queries("df['temperature']", "df[['time', 'temperature']]")
        assert result.matches is False


class TestExperimentConfig:
    """Tests for experiment configuration."""

    def test_config_creation(self):
        config = ExperimentConfig(
            model="gemini-2.5-flash",
            enrichment=True,
            questions_path="data/evaluation/questions.yaml",
            output_path="results/test.yaml",
        )
        assert config.model == "gemini-2.5-flash"
        assert config.provider == "gemini"  # auto-derived
        assert config.enrichment is True

    def test_provider_derived_from_gemini(self):
        config = ExperimentConfig(
            model="gemini-2.5-flash",
            enrichment=True,
            questions_path="q.yaml",
            output_path="o.yaml",
        )
        assert config.provider == "gemini"

    def test_provider_derived_from_gpt(self):
        config = ExperimentConfig(
            model="gpt-5.4-mini",
            enrichment=False,
            questions_path="q.yaml",
            output_path="o.yaml",
        )
        assert config.provider == "openai"


class TestExperimentResult:
    """Tests for ExperimentResult metrics."""

    def test_empty_results(self):
        from nl2vis_bench.validator.experiment import ExperimentResult

        result = ExperimentResult()
        assert result.ground_truth_match_rate == 0.0
        assert result.execution_success_rate == 0.0
        assert result.viz_accuracy == 0.0
        assert result.columns_accuracy == 0.0

    def test_accuracy_calculations(self):
        from nl2vis_bench.validator.experiment import QuestionResult, ExperimentResult

        result = ExperimentResult(
            results=[
                QuestionResult(
                    question_id="q1",
                    question="Q1",
                    category="test",
                    ground_truth_match=True,
                    execution_success=True,
                    viz_correct=True,
                    columns_correct=True,
                ),
                QuestionResult(
                    question_id="q2",
                    question="Q2",
                    category="test",
                    ground_truth_match=False,
                    execution_success=True,
                    viz_correct=False,
                    columns_correct=False,
                ),
            ]
        )
        assert result.ground_truth_match_rate == 0.5
        assert result.execution_success_rate == 1.0
        assert result.viz_accuracy == 0.5
        assert result.columns_accuracy == 0.5

    def test_manual_review_ids(self):
        from nl2vis_bench.validator.experiment import QuestionResult, ExperimentResult

        result = ExperimentResult(
            results=[
                QuestionResult(
                    question_id="q1",
                    question="Q1",
                    category="test",
                    needs_manual_review=True,
                ),
                QuestionResult(
                    question_id="q2",
                    question="Q2",
                    category="test",
                    needs_manual_review=False,
                ),
                QuestionResult(
                    question_id="q3",
                    question="Q3",
                    category="test",
                    needs_manual_review=True,
                ),
            ]
        )
        assert result.manual_review_ids == ["q1", "q3"]


class TestExperimentConfigErrors:
    """Tests for error handling."""

    def test_unknown_model_raises(self):
        config = ExperimentConfig(
            model="unknown-model",
            enrichment=True,
            questions_path="q.yaml",
            output_path="o.yaml",
        )
        with pytest.raises(ValueError, match="Unknown model provider"):
            _ = config.provider  # Accessing property triggers error


class TestExperimentRunner:
    """Tests for experiment runner."""

    @pytest.fixture
    def mock_llm(self):
        from unittest.mock import Mock
        mock = Mock()
        mock.complete.return_value = '{"query": "df[[\'time\', \'NEE_CUT_REF\']]", "columns_used": ["time", "NEE_CUT_REF"], "explanation": "Time series"}'
        mock.model_name.return_value = "gemini/gemini-2.5-flash"
        return mock

    @pytest.fixture
    def mini_questions_path(self, tmp_path):
        content = """
version: "2.0"
dataset: "test-dataset"
questions:
  - id: temp_001
    question: "Show NEE over time"
    category: "temporal"
    expected_query: "df[['time', 'NEE_CUT_REF']]"
    expected_viz: "line"
    expected_x: "time"
    expected_y: "NEE_CUT_REF"
    expected_columns: ["time", "NEE_CUT_REF"]
"""
        path = tmp_path / "questions.yaml"
        path.write_text(content)
        return path

    def test_runner_evaluates_question(self, mock_llm, mini_questions_path, tmp_path):
        import pandas as pd
        from nl2vis_bench.models import DatasetSchema, ColumnInfo
        from nl2vis_bench.validator.experiment import ExperimentRunner

        schema = DatasetSchema(
            name="test",
            file_type="csv",
            columns=[
                ColumnInfo(name="time", dtype="datetime"),
                ColumnInfo(name="NEE_CUT_REF", dtype="float"),
            ]
        )

        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10),
            "NEE_CUT_REF": range(10),
        })

        runner = ExperimentRunner(llm_provider=mock_llm)
        result = runner.run_single_question(
            question={
                "id": "temp_001",
                "question": "Show NEE over time",
                "category": "temporal",
                "expected_query": "df[['time', 'NEE_CUT_REF']]",
                "expected_viz": "line",
                "expected_x": "time",
                "expected_y": "NEE_CUT_REF",
            },
            schema=schema,
            df=df,
            use_enrichment=True,
        )

        assert result.question_id == "temp_001"
        assert result.ground_truth_match is True
        assert result.execution_success is True


class TestReportGeneration:
    """Tests for YAML report generation."""

    def test_generate_report_structure(self):
        from nl2vis_bench.validator.report import generate_report
        from nl2vis_bench.validator.experiment import ExperimentResult, QuestionResult

        result = ExperimentResult(
            metadata={
                "model": "gemini/gemini-2.5-flash",
                "enrichment": True,
                "total_questions": 2,
            },
            results=[
                QuestionResult(
                    question_id="test_001",
                    question="Test question 1",
                    category="temporal",
                    ground_truth_match=True,
                    execution_success=True,
                    viz_type_correct=True,
                    axes_correct=True,
                    viz_correct=True,
                    columns_correct=True,
                ),
                QuestionResult(
                    question_id="test_002",
                    question="Test question 2",
                    category="distribution",
                    ground_truth_match=False,
                    execution_success=True,
                    needs_manual_review=True,
                    columns_correct=False,
                ),
            ]
        )

        report = generate_report(result)

        assert "metadata" in report
        assert "summary" in report
        assert "results" in report

        # Verify all summary metrics
        assert report["summary"]["ground_truth_match_rate"] == 0.5
        assert report["summary"]["execution_success_rate"] == 1.0
        assert report["summary"]["viz_accuracy"] == 0.5
        assert report["summary"]["columns_accuracy"] == 0.5
        assert report["summary"]["manual_review_needed"]["count"] == 1
        assert report["summary"]["manual_review_needed"]["question_ids"] == ["test_002"]

        # Verify results structure
        assert len(report["results"]) == 2
        assert report["results"][0]["id"] == "test_001"

    def test_save_report_creates_file(self, tmp_path):
        """Test that save_report creates YAML file."""
        from nl2vis_bench.validator.report import save_report
        from nl2vis_bench.validator.experiment import ExperimentResult, QuestionResult

        result = ExperimentResult(
            metadata={"model": "test"},
            results=[
                QuestionResult(question_id="q1", question="Q1", category="test")
            ]
        )

        output_file = tmp_path / "reports" / "test_report.yaml"
        save_report(result, output_file)

        assert output_file.exists()
        import yaml
        with open(output_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded["metadata"]["model"] == "test"
        assert len(loaded["results"]) == 1
