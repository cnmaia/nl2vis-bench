"""CLI for running validation experiments."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from nl2vis_bench.models import DatasetSchema, ColumnInfo
from nl2vis_bench.validator import ExperimentConfig, ExperimentRunner, save_report
from nl2vis_bench.validator.stats import aggregate_results


def create_llm_provider(model: str):
    """Create appropriate LLM provider based on model name."""
    if model.startswith("gemini"):
        from nl2vis_bench.llm.gemini import GeminiProvider

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return GeminiProvider(api_key=api_key, model=model)
    elif model.startswith("gpt"):
        from nl2vis_bench.llm.openai_provider import OpenAIProvider

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown model: {model}")


def load_schema_from_csv(
    csv_path: str | Path, dataset_name: str = "unknown"
) -> DatasetSchema:
    """Load dataset schema from column_metadata.csv."""
    df = pd.read_csv(csv_path)

    if "column_name" not in df.columns:
        raise ValueError(
            f"CSV must have 'column_name' column. Found: {list(df.columns)}"
        )

    columns = []
    for _, row in df.iterrows():
        col = ColumnInfo(
            name=row["column_name"],
            dtype=row.get("dtype", "float"),
            description=row.get("description"),
            min_value=row.get("min_value"),
            max_value=row.get("max_value"),
            mean_value=row.get("mean_value"),
        )
        columns.append(col)
    return DatasetSchema(
        name=dataset_name,
        file_type="csv",
        columns=columns,
    )


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = argparse.ArgumentParser(description="Run NL2Vis validation experiment")
    parser.add_argument(
        "--model", required=True, help="LLM model (gemini-2.5-flash or gpt-5.4-mini)"
    )
    parser.add_argument(
        "--enrichment",
        type=lambda x: x.lower() == "true",
        required=True,
        help="Use semantic enrichment (true/false)",
    )
    parser.add_argument(
        "--questions",
        default="data/evaluation/questions.yaml",
        help="Path to questions YAML file",
    )
    parser.add_argument(
        "--schema",
        default="data/column_metadata.csv",
        help="Path to column metadata CSV",
    )
    parser.add_argument(
        "--data",
        default="data/sample.parquet",
        help="Path to sample data file (parquet/csv)",
    )
    parser.add_argument("--output", required=True, help="Output path for results YAML")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum questions to evaluate (for testing)",
    )
    parser.add_argument(
        "--dataset-name",
        default="NEE-METEORS-AMAZON_BRSa1_2002_2011",
        help="Dataset name to use in schema",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of experiment runs for statistical analysis (default: 1)",
    )
    parser.add_argument(
        "--detailed-logs",
        action="store_true",
        help="Save detailed per-question logs for analysis"
    )

    args = parser.parse_args()

    try:
        print("Running experiment:")
        print(f"  Model: {args.model}")
        print(f"  Enrichment: {args.enrichment}")
        print(f"  Questions: {args.questions}")
        print(f"  Output: {args.output}")
        print(f"  Runs: {args.runs}")

        # Create config
        config = ExperimentConfig(
            model=args.model,
            enrichment=args.enrichment,
            questions_path=args.questions,
            output_path=args.output,
        )

        # Load schema
        print(f"\nLoading schema from {args.schema}...")
        schema = load_schema_from_csv(args.schema, dataset_name=args.dataset_name)
        print(f"  Loaded {len(schema.columns)} columns")

        # Load data
        print(f"\nLoading data from {args.data}...")
        if args.data.endswith(".parquet"):
            df = pd.read_parquet(args.data)
        else:
            # Parse 'time' column as datetime if it exists
            df = pd.read_csv(args.data, parse_dates=['time'])
        print(f"  Loaded {len(df)} rows")

        # Create LLM provider
        print("\nInitializing LLM provider...")
        llm = create_llm_provider(args.model)

        # Run experiment(s)
        runner = ExperimentRunner(llm_provider=llm, detailed_logging=args.detailed_logs)
        all_run_results = []
        all_run_metrics = []

        for run_num in range(1, args.runs + 1):
            print(f"\n=== Run {run_num}/{args.runs} ===")
            result = runner.run(
                config=config,
                schema=schema,
                df=df,
                max_questions=args.max_questions,
            )
            all_run_results.append(result)

            # Save detailed logs if enabled
            if args.detailed_logs and runner.logger:
                log_filename = f"{args.model}_{args.enrichment}_run{run_num}.jsonl"
                saved_path = runner.logger.save(log_filename)
                print(f"  Detailed logs saved to: {saved_path}")
                runner.logger.clear()

            # Extract metrics for this run (convert to native Python float for YAML)
            metrics = {
                "execution_success_rate": float(result.execution_success_rate),
                "viz_type_accuracy": float(result.viz_type_accuracy),
                "axes_accuracy": float(result.axes_accuracy),
                "viz_accuracy": float(result.viz_accuracy),
                "ground_truth_match_rate": float(result.ground_truth_match_rate),
            }
            all_run_metrics.append(metrics)

            print(f"  Execution: {result.execution_success_rate:.1%}")
            print(f"  Viz accuracy: {result.viz_accuracy:.1%}")

        # Save results
        print(f"\nSaving results to {args.output}...")
        if args.runs > 1:
            # Multi-run: aggregate and save both individual and aggregated
            aggregated = aggregate_results(all_run_metrics)
            output_data = {
                "metadata": {
                    "model": args.model,
                    "enrichment": args.enrichment,
                    "runs": args.runs,
                },
                "aggregated": aggregated,
                "individual_runs": all_run_metrics,
            }
            with open(args.output, "w") as f:
                yaml.safe_dump(output_data, f, default_flow_style=False)

            # Print aggregated summary
            print("\n=== Aggregated Results ===")
            for metric, data in aggregated.items():
                mean = data["mean"]
                std = data["std"]
                ci = data["ci_95"]
                print(f"{metric}: {mean:.1%} ± {std:.1%} (95% CI: [{ci[0]:.1%}, {ci[1]:.1%}])")
        else:
            # Single run: save as before
            result = all_run_results[0]
            save_report(result, args.output)

            # Print summary
            print("\n=== Results ===")
            print(f"Ground truth match rate: {result.ground_truth_match_rate:.1%}")
            print(f"Execution success rate: {result.execution_success_rate:.1%}")
            print(f"Viz type accuracy: {result.viz_type_accuracy:.1%}")
            print(f"Axes accuracy: {result.axes_accuracy:.1%}")
            print(f"Viz accuracy: {result.viz_accuracy:.1%}")
            if result.manual_review_ids:
                print(f"\nNeeds manual review ({len(result.manual_review_ids)}):")
                for qid in result.manual_review_ids[:5]:
                    print(f"  - {qid}")
                if len(result.manual_review_ids) > 5:
                    print(f"  ... and {len(result.manual_review_ids) - 5} more")

        return 0  # Success

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
