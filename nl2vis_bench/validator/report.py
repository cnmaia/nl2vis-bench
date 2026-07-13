"""YAML report generation for experiment results."""
from pathlib import Path
from typing import Any

import yaml

from .experiment import ExperimentResult


def generate_report(result: ExperimentResult) -> dict[str, Any]:
    """Generate report dictionary from experiment result."""
    return {
        "metadata": result.metadata,
        "summary": {
            "ground_truth_match_rate": round(result.ground_truth_match_rate, 2),
            "columns_accuracy": round(result.columns_accuracy, 2),
            "execution_success_rate": round(result.execution_success_rate, 2),
            "viz_type_accuracy": round(result.viz_type_accuracy, 2),
            "axes_accuracy": round(result.axes_accuracy, 2),
            "viz_accuracy": round(result.viz_accuracy, 2),
            "manual_review_needed": {
                "count": len(result.manual_review_ids),
                "question_ids": result.manual_review_ids,
            },
        },
        "results": [
            {
                "id": r.question_id,
                "question": r.question,
                "category": r.category,
                "generated_query": r.generated_query,
                "expected_query": r.expected_query,
                "ground_truth_match": r.ground_truth_match,
                "columns_correct": r.columns_correct,
                "execution_success": r.execution_success,
                "selected_viz": r.selected_viz,
                "expected_viz": r.expected_viz,
                "viz_type_correct": r.viz_type_correct,
                "selected_x": r.selected_x,
                "selected_y": r.selected_y,
                "expected_x": r.expected_x,
                "expected_y": r.expected_y,
                "axes_correct": r.axes_correct,
                "viz_correct": r.viz_correct,
                "needs_manual_review": r.needs_manual_review,
                "latency_ms": round(r.latency_ms, 1),
                "error": r.error,
            }
            for r in result.results
        ],
    }


def save_report(result: ExperimentResult, output_path: str | Path) -> None:
    """Save experiment result as YAML file."""
    report = generate_report(result)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
