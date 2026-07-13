"""Analyze experiment results and compute t-test for statistical significance."""
import yaml
from nl2vis_bench.validator.stats import compare_conditions


def load_results(filepath):
    """Load results from YAML file."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def main():
    # Load all results (using the new files with datetime parsing fix)
    gpt_with = load_results("results/gpt_with_enrichment_5runs_new.yaml")
    gpt_without = load_results("results/gpt_no_enrichment_5runs_new.yaml")
    gemini_with = load_results("results/gemini_with_enrichment_5runs_new.yaml")
    gemini_without = load_results("results/gemini_no_enrichment_5runs_new.yaml")

    print("=" * 60)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("=" * 60)

    # Print summary of results
    print("\n1. EXECUTION SUCCESS RATE")
    print("-" * 40)
    print(f"GPT + enrichment:    {gpt_with['aggregated']['execution_success_rate']['mean']*100:.1f}% "
          f"± {gpt_with['aggregated']['execution_success_rate']['std']*100:.1f}%")
    print(f"GPT no enrichment:   {gpt_without['aggregated']['execution_success_rate']['mean']*100:.1f}% "
          f"± {gpt_without['aggregated']['execution_success_rate']['std']*100:.1f}%")
    print(f"Gemini + enrichment: {gemini_with['aggregated']['execution_success_rate']['mean']*100:.1f}% "
          f"± {gemini_with['aggregated']['execution_success_rate']['std']*100:.1f}%")
    print(f"Gemini no enrichment:{gemini_without['aggregated']['execution_success_rate']['mean']*100:.1f}% "
          f"± {gemini_without['aggregated']['execution_success_rate']['std']*100:.1f}%")

    print("\n2. VISUALIZATION ACCURACY")
    print("-" * 40)
    print(f"GPT + enrichment:    {gpt_with['aggregated']['viz_accuracy']['mean']*100:.1f}% "
          f"± {gpt_with['aggregated']['viz_accuracy']['std']*100:.1f}%")
    print(f"GPT no enrichment:   {gpt_without['aggregated']['viz_accuracy']['mean']*100:.1f}% "
          f"± {gpt_without['aggregated']['viz_accuracy']['std']*100:.1f}%")
    print(f"Gemini + enrichment: {gemini_with['aggregated']['viz_accuracy']['mean']*100:.1f}% "
          f"± {gemini_with['aggregated']['viz_accuracy']['std']*100:.1f}%")
    print(f"Gemini no enrichment:{gemini_without['aggregated']['viz_accuracy']['mean']*100:.1f}% "
          f"± {gemini_without['aggregated']['viz_accuracy']['std']*100:.1f}%")

    # Extract execution success rates for t-test
    gpt_with_exec = gpt_with["aggregated"]["execution_success_rate"]["runs"]
    gpt_without_exec = gpt_without["aggregated"]["execution_success_rate"]["runs"]
    gemini_with_exec = gemini_with["aggregated"]["execution_success_rate"]["runs"]
    gemini_without_exec = gemini_without["aggregated"]["execution_success_rate"]["runs"]

    print("\n" + "=" * 60)
    print("3. T-TEST FOR EXECUTION SUCCESS RATE")
    print("=" * 60)

    # T-test for GPT
    gpt_ttest = compare_conditions(gpt_with_exec, gpt_without_exec)
    print(f"\nGPT Enrichment Effect (Execution):")
    print(f"  With:    {[f'{x*100:.1f}%' for x in gpt_with_exec]}")
    print(f"  Without: {[f'{x*100:.1f}%' for x in gpt_without_exec]}")
    print(f"  t-statistic: {gpt_ttest['t_statistic']:.3f}")
    print(f"  p-value: {gpt_ttest['p_value']:.6f}")
    print(f"  Significant (p < 0.05): {gpt_ttest['significant']}")
    print(f"  Degrees of freedom: 8 (n1 + n2 - 2)")

    # Same for Gemini
    gemini_ttest = compare_conditions(gemini_with_exec, gemini_without_exec)
    print(f"\nGemini Enrichment Effect (Execution):")
    print(f"  With:    {[f'{x*100:.1f}%' for x in gemini_with_exec]}")
    print(f"  Without: {[f'{x*100:.1f}%' for x in gemini_without_exec]}")
    print(f"  t-statistic: {gemini_ttest['t_statistic']:.3f}")
    print(f"  p-value: {gemini_ttest['p_value']:.6f}")
    print(f"  Significant (p < 0.05): {gemini_ttest['significant']}")

    # Extract visualization accuracy for t-test
    gpt_with_viz = gpt_with["aggregated"]["viz_accuracy"]["runs"]
    gpt_without_viz = gpt_without["aggregated"]["viz_accuracy"]["runs"]
    gemini_with_viz = gemini_with["aggregated"]["viz_accuracy"]["runs"]
    gemini_without_viz = gemini_without["aggregated"]["viz_accuracy"]["runs"]

    print("\n" + "=" * 60)
    print("4. T-TEST FOR VISUALIZATION ACCURACY")
    print("=" * 60)

    # T-test for GPT viz accuracy
    gpt_viz_ttest = compare_conditions(gpt_with_viz, gpt_without_viz)
    print(f"\nGPT Enrichment Effect (Viz Accuracy):")
    print(f"  With:    {[f'{x*100:.1f}%' for x in gpt_with_viz]}")
    print(f"  Without: {[f'{x*100:.1f}%' for x in gpt_without_viz]}")
    print(f"  t-statistic: {gpt_viz_ttest['t_statistic']:.3f}")
    print(f"  p-value: {gpt_viz_ttest['p_value']:.6f}")
    print(f"  Significant (p < 0.05): {gpt_viz_ttest['significant']}")

    # Same for Gemini
    gemini_viz_ttest = compare_conditions(gemini_with_viz, gemini_without_viz)
    print(f"\nGemini Enrichment Effect (Viz Accuracy):")
    print(f"  With:    {[f'{x*100:.1f}%' for x in gemini_with_viz]}")
    print(f"  Without: {[f'{x*100:.1f}%' for x in gemini_without_viz]}")
    print(f"  t-statistic: {gemini_viz_ttest['t_statistic']:.3f}")
    print(f"  p-value: {gemini_viz_ttest['p_value']:.6f}")
    print(f"  Significant (p < 0.05): {gemini_viz_ttest['significant']}")

    # Summary for paper
    print("\n" + "=" * 60)
    print("5. SUMMARY FOR PAPER")
    print("=" * 60)

    gpt_exec_diff = (gpt_with["aggregated"]["execution_success_rate"]["mean"] -
                     gpt_without["aggregated"]["execution_success_rate"]["mean"]) * 100
    gpt_viz_diff = (gpt_with["aggregated"]["viz_accuracy"]["mean"] -
                    gpt_without["aggregated"]["viz_accuracy"]["mean"]) * 100

    gemini_exec_diff = (gemini_with["aggregated"]["execution_success_rate"]["mean"] -
                        gemini_without["aggregated"]["execution_success_rate"]["mean"]) * 100
    gemini_viz_diff = (gemini_with["aggregated"]["viz_accuracy"]["mean"] -
                       gemini_without["aggregated"]["viz_accuracy"]["mean"]) * 100

    print(f"\nGPT-4o-mini:")
    print(f"  Execution: {gpt_without['aggregated']['execution_success_rate']['mean']*100:.1f}% -> "
          f"{gpt_with['aggregated']['execution_success_rate']['mean']*100:.1f}% "
          f"({gpt_exec_diff:+.1f} pp)")
    print(f"  Viz Acc:   {gpt_without['aggregated']['viz_accuracy']['mean']*100:.1f}% -> "
          f"{gpt_with['aggregated']['viz_accuracy']['mean']*100:.1f}% "
          f"({gpt_viz_diff:+.1f} pp)")
    print(f"  Exec t-test: t(8) = {gpt_ttest['t_statistic']:.2f}, p = {gpt_ttest['p_value']:.4f}")
    print(f"  Viz t-test:  t(8) = {gpt_viz_ttest['t_statistic']:.2f}, p = {gpt_viz_ttest['p_value']:.4f}")

    print(f"\nGemini 2.5 Flash:")
    print(f"  Execution: {gemini_without['aggregated']['execution_success_rate']['mean']*100:.1f}% -> "
          f"{gemini_with['aggregated']['execution_success_rate']['mean']*100:.1f}% "
          f"({gemini_exec_diff:+.1f} pp)")
    print(f"  Viz Acc:   {gemini_without['aggregated']['viz_accuracy']['mean']*100:.1f}% -> "
          f"{gemini_with['aggregated']['viz_accuracy']['mean']*100:.1f}% "
          f"({gemini_viz_diff:+.1f} pp)")
    print(f"  Exec t-test: t(8) = {gemini_ttest['t_statistic']:.2f}, p = {gemini_ttest['p_value']:.4f}")
    print(f"  Viz t-test:  t(8) = {gemini_viz_ttest['t_statistic']:.2f}, p = {gemini_viz_ttest['p_value']:.4f}")


if __name__ == "__main__":
    main()
