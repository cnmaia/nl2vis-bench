from nl2vis_bench.validator.stats import compare_conditions


def test_compare_conditions_significant_difference():
    """T-test should detect significant difference between conditions."""
    # Simulated results: clear difference
    with_enrichment = [0.80, 0.82, 0.84, 0.81, 0.83]  # mean ~0.82
    without_enrichment = [0.60, 0.62, 0.58, 0.61, 0.59]  # mean ~0.60

    result = compare_conditions(with_enrichment, without_enrichment)

    assert "t_statistic" in result
    assert "p_value" in result
    assert "significant" in result
    assert result["significant"] is True, f"Expected significant, p={result['p_value']}"
    assert result["p_value"] < 0.05


def test_compare_conditions_no_significant_difference():
    """T-test should not flag similar distributions as significant."""
    condition_a = [0.80, 0.81, 0.79, 0.80, 0.80]
    condition_b = [0.79, 0.80, 0.81, 0.80, 0.79]

    result = compare_conditions(condition_a, condition_b)

    assert (
        result["significant"] is False
    ), f"Should not be significant, p={result['p_value']}"
