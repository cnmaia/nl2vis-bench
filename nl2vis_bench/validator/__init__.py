"""Validator module for SBBD'26 paper experiments."""
from .ground_truth import normalize_query, compare_queries, QueryMatch
from .experiment import ExperimentConfig, QuestionResult, ExperimentResult, ExperimentRunner
from .report import generate_report, save_report

__all__ = [
    "normalize_query",
    "compare_queries",
    "QueryMatch",
    "ExperimentConfig",
    "QuestionResult",
    "ExperimentResult",
    "ExperimentRunner",
    "generate_report",
    "save_report",
]
