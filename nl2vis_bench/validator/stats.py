"""Statistical utilities for multi-run experiments."""

import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from dataclasses import dataclass


@dataclass
class AggregatedMetric:
    """Statistical summary of a metric across multiple runs."""

    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float
    runs: list[float]

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "ci_95": [self.ci_95_lower, self.ci_95_upper],
            "runs": self.runs,
        }


def aggregate_metric(values: list[float]) -> AggregatedMetric:
    """Compute mean, std, and 95% CI for a list of values.

    Uses t-distribution for CI calculation (appropriate for small samples).
    """
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # 95% CI using t-distribution
    if n > 1 and std > 0:
        t_critical = stats.t.ppf(0.975, df=n - 1)
        margin = t_critical * (std / np.sqrt(n))
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = ci_upper = mean

    return AggregatedMetric(
        mean=float(round(mean, 4)),
        std=float(round(std, 4)),
        ci_95_lower=float(round(ci_lower, 4)),
        ci_95_upper=float(round(ci_upper, 4)),
        runs=[float(round(v, 4)) for v in values],
    )


def aggregate_results(run_results: list[dict]) -> dict:
    """Aggregate metrics across multiple experiment runs.

    Args:
        run_results: List of result dicts, each containing metric values

    Returns:
        Dict with aggregated metrics (all values are native Python types)
    """
    metrics = [
        "execution_success_rate",
        "viz_type_accuracy",
        "axes_accuracy",
        "viz_accuracy",
        "ground_truth_match_rate",
    ]

    aggregated = {}
    for metric in metrics:
        # Convert to native Python float to avoid YAML serialization issues
        values = [float(r.get(metric, 0.0)) for r in run_results]
        aggregated[metric] = aggregate_metric(values).to_dict()

    return aggregated


def compare_conditions(
    with_enrichment: list[float], without_enrichment: list[float]
) -> dict:
    """Independent samples t-test between two conditions.

    Args:
        with_enrichment: List of metric values with enrichment
        without_enrichment: List of metric values without enrichment

    Returns:
        Dict with t_statistic, p_value, and significant (bool)
    """
    t_stat, p_value = ttest_ind(with_enrichment, without_enrichment)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
