import pytest
import tempfile
import json
from pathlib import Path
from nl2vis_bench.validator.detailed_logger import DetailedLogger, QuestionLog


def test_detailed_logger_saves_jsonl():
    """Logger should save entries as JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = DetailedLogger(output_dir=tmpdir)

        entry = QuestionLog(
            question_id="q01",
            question="Show NEE over time",
            generated_code="df.plot(x='time', y='NEE_CUT_REF')",
            execution_success=True,
            execution_error=None,
            expected_viz_type="line",
            actual_viz_type="line",
            expected_columns=["time", "NEE_CUT_REF"],
            actual_columns=["time", "NEE_CUT_REF"],
            viz_type_correct=True,
            axes_correct=True,
            overall_correct=True
        )
        logger.log(entry)
        filepath = logger.save("test_run.jsonl")

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.loads(f.readline())
        assert loaded["question_id"] == "q01"
        assert loaded["execution_success"] is True
