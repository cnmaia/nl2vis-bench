"""Detailed per-question logging for experiment analysis."""
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class QuestionLog:
    """Detailed log entry for a single question."""
    question_id: str
    question: str
    generated_code: str
    execution_success: bool
    execution_error: Optional[str]
    expected_viz_type: str
    actual_viz_type: Optional[str]
    expected_columns: list[str]
    actual_columns: list[str]
    viz_type_correct: bool
    axes_correct: bool
    overall_correct: bool


class DetailedLogger:
    """Logs detailed per-question results to JSONL files."""

    def __init__(self, output_dir: str = "results/detailed_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[QuestionLog] = []

    def log(self, entry: QuestionLog):
        """Add a log entry."""
        self.entries.append(entry)

    def save(self, filename: str):
        """Save all entries to a JSONL file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            for entry in self.entries:
                f.write(json.dumps(asdict(entry)) + '\n')
        return filepath

    def clear(self):
        """Clear entries for next run."""
        self.entries = []
