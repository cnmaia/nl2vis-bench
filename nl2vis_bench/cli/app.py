"""Main CLI application for scientific data catalogs-Vis."""
import json
import os
from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID
import typer
import pandas as pd

from nl2vis_bench import __version__
from nl2vis_bench.catalog import CatalogRepository
from nl2vis_bench.catalog.postgres_repository import PostgresCatalogRepository
from nl2vis_bench.gateway import MinIOGateway
from nl2vis_bench.indexers import HandlerRegistry, XLSXHandler, CSVHandler, CampbellCR6Handler
from nl2vis_bench.llm import GeminiProvider, OpenAIProvider
from nl2vis_bench.translation import NLTranslator
from nl2vis_bench.execution import QueryExecutor
from nl2vis_bench.visualization import VizSelector, PluginRegistry
from nl2vis_bench.models import VizSpec, DatasetSchema


def _enrich_viz_labels(
    viz_spec: VizSpec,
    schema: DatasetSchema,
    question: str,
    explanation: str,
) -> VizSpec:
    """Enrich VizSpec with user-friendly labels from column descriptions."""
    # Build column name to description mapping (from LLM-generated metadata)
    col_descriptions = {col.name: col.description for col in schema.columns if col.description}

    # Title is the user's question
    viz_spec.title = question.capitalize().rstrip("?").rstrip(".")

    # Subtitle explains what's shown, using column description if available
    if viz_spec.y and viz_spec.y in col_descriptions:
        viz_spec.subtitle = col_descriptions[viz_spec.y]
    elif explanation:
        viz_spec.subtitle = explanation[:120] + "..." if len(explanation) > 120 else explanation

    # Y-axis label stays as column name (set by the plugin)

    return viz_spec


app = typer.Typer(
    name="nl2vis-bench",
    help="Natural language data visualization for scientific data catalogs environmental data.",
    add_completion=False,
)

# Config directory
CONFIG_DIR = Path.home() / ".nl2vis-bench"
CONFIG_FILE = CONFIG_DIR / "config.json"
CATALOG_DB = CONFIG_DIR / "catalog.db"


def get_config() -> dict:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(config: dict) -> None:
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_catalog() -> CatalogRepository:
    """Get local catalog repository instance (fallback for standalone mode)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CatalogRepository(str(CATALOG_DB))


def get_postgres_catalog() -> PostgresCatalogRepository | None:
    """Get PostgreSQL catalog repository if DATABASE_URL is configured."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    return PostgresCatalogRepository(db_url)


def get_minio_gateway() -> MinIOGateway | None:
    """Get MinIO gateway if environment variables are configured."""
    endpoint = os.environ.get("MINIO_URL")
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")
    bucket = os.environ.get("MINIO_BUCKET", "data")
    secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"

    if not all([endpoint, access_key, secret_key]):
        return None

    return MinIOGateway(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        secure=secure,
    )


def get_handler_registry() -> HandlerRegistry:
    """Get handler registry with all registered handlers."""
    registry = HandlerRegistry()
    registry.register(XLSXHandler())
    registry.register(CSVHandler())
    registry.register(CampbellCR6Handler())
    return registry


def get_llm_provider(config: dict):
    """Get LLM provider based on configuration."""
    provider = config.get("llm_provider", "gemini")

    if provider == "gemini":
        api_key = config.get("gemini_api_key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            typer.echo("Error: Gemini API key not configured. Run 'nl2vis-bench config' first.", err=True)
            raise typer.Exit(1)
        return GeminiProvider(api_key=api_key)

    elif provider == "openai":
        api_key = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            typer.echo("Error: OpenAI API key not configured. Run 'nl2vis-bench config' first.", err=True)
            raise typer.Exit(1)
        return OpenAIProvider(api_key=api_key)

    else:
        typer.echo(f"Error: Unknown LLM provider: {provider}", err=True)
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    typer.echo(f"nl2vis-bench version {__version__}")


@app.command()
def config(
    provider: Annotated[Optional[str], typer.Option("--provider", "-p", help="LLM provider (gemini/openai)")] = None,
    gemini_key: Annotated[Optional[str], typer.Option("--gemini-key", help="Gemini API key")] = None,
    openai_key: Annotated[Optional[str], typer.Option("--openai-key", help="OpenAI API key")] = None,
    show: Annotated[bool, typer.Option("--show", help="Show current configuration")] = False,
):
    """Configure NL2Vis settings."""
    current_config = get_config()

    if show:
        if not current_config:
            typer.echo("No configuration found.")
        else:
            typer.echo("Current configuration:")
            typer.echo(f"  LLM Provider: {current_config.get('llm_provider', 'not set')}")
            typer.echo(f"  Gemini API Key: {'***' if current_config.get('gemini_api_key') else 'not set'}")
            typer.echo(f"  OpenAI API Key: {'***' if current_config.get('openai_api_key') else 'not set'}")
        return

    if provider:
        if provider not in ["gemini", "openai"]:
            typer.echo(f"Error: Invalid provider '{provider}'. Use 'gemini' or 'openai'.", err=True)
            raise typer.Exit(1)
        current_config["llm_provider"] = provider

    if gemini_key:
        current_config["gemini_api_key"] = gemini_key

    if openai_key:
        current_config["openai_api_key"] = openai_key

    if provider or gemini_key or openai_key:
        save_config(current_config)
        typer.echo("Configuration saved.")
    else:
        typer.echo("No changes made. Use --help for options.")


@app.command()
def index(
    path: Annotated[Path, typer.Argument(help="Path to file or directory to index")],
    name: Annotated[Optional[str], typer.Option("--name", "-n", help="Dataset name (defaults to filename)")] = None,
):
    """Index a dataset file or directory."""
    if not path.exists():
        typer.echo(f"Error: Path does not exist: {path}", err=True)
        raise typer.Exit(1)

    registry = get_handler_registry()
    catalog = get_catalog()

    # Get file extension
    if path.is_file():
        ext = path.suffix.lower().lstrip(".")
        handler = registry.get_handler(str(path))

        if handler is None:
            typer.echo(f"Error: No handler for file type '.{ext}'", err=True)
            typer.echo(f"Supported types: {', '.join(registry.list_supported())}")
            raise typer.Exit(1)

        # Index the file
        typer.echo(f"Indexing {path}...")
        schema = handler.extract_schema(str(path))
        schema.file_paths = [str(path)]

        if name:
            schema.name = name

        catalog.save_dataset(schema)
        typer.echo(f"Indexed dataset '{schema.name}' with {len(schema.columns)} columns.")
    else:
        typer.echo("Directory indexing not yet implemented.", err=True)
        raise typer.Exit(1)


@app.command("list")
def list_datasets():
    """List all indexed datasets."""
    catalog = get_catalog()
    datasets = catalog.list_datasets()

    if not datasets:
        typer.echo("No datasets indexed yet. Use 'nl2vis-bench index' to add one.")
        return

    typer.echo(f"Found {len(datasets)} dataset(s):\n")

    for ds in datasets:
        typer.echo(f"  {ds.name}")
        typer.echo(f"    Type: {ds.file_type}")
        typer.echo(f"    Columns: {len(ds.columns)}")
        typer.echo(f"    Files: {ds.file_count}")
        typer.echo()


@app.command()
def query(
    dataset: Annotated[str, typer.Argument(help="Dataset name to query")],
    question: Annotated[str, typer.Argument(help="Natural language question")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output path for visualization")] = None,
    no_viz: Annotated[bool, typer.Option("--no-viz", help="Skip visualization, show data only")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output")] = False,
):
    """Query a dataset using natural language."""
    config = get_config()
    catalog = get_catalog()

    # Get dataset schema
    schema = catalog.get_dataset(dataset)
    if schema is None:
        typer.echo(f"Error: Dataset '{dataset}' not found.", err=True)
        typer.echo("Use 'nl2vis-bench list' to see available datasets.")
        raise typer.Exit(1)

    # Load the data
    if not schema.file_paths:
        typer.echo("Error: Dataset has no associated files.", err=True)
        raise typer.Exit(1)

    file_path = schema.file_paths[0]
    registry = get_handler_registry()
    handler = registry.get_handler(file_path)

    if handler is None:
        typer.echo(f"Error: No handler for file type '{schema.file_type}'", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading data from {file_path}...")
    df = handler.load(file_path)

    # Translate question to query
    typer.echo("Translating question...")
    llm = get_llm_provider(config)
    translator = NLTranslator(llm_provider=llm)

    try:
        result = translator.translate(question, schema)
    except ValueError as e:
        typer.echo(f"Error translating question: {e}", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"\nGenerated query: {result.query}")
        typer.echo(f"Columns used: {result.columns_used}")
        typer.echo(f"Explanation: {result.explanation}")
        typer.echo(f"Latency: {result.latency_ms:.0f}ms")

    # Execute query
    typer.echo("Executing query...")
    executor = QueryExecutor()
    exec_result = executor.execute(df, result.query)

    if not exec_result.success:
        typer.echo(f"Error executing query: {exec_result.error_message}", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Execution time: {exec_result.execution_time_ms:.0f}ms")
        typer.echo(f"Result rows: {exec_result.profile.row_count}")

    # Show data preview
    typer.echo("\nResult:")
    typer.echo(exec_result.data.to_string(max_rows=10))

    if no_viz:
        return

    # Select visualization
    typer.echo("\nSelecting visualization...")
    selector = VizSelector(llm_provider=llm)
    viz_spec = selector.select(exec_result.profile, question)

    if verbose:
        typer.echo(f"Visualization type: {viz_spec.type}")
        typer.echo(f"X: {viz_spec.x}, Y: {viz_spec.y}")
        typer.echo(f"Selected by: {viz_spec.selected_by}")

    # Render visualization
    plugin_registry = PluginRegistry.default()
    plugin = plugin_registry.get(viz_spec.type)

    if plugin is None:
        typer.echo(f"Error: No plugin for visualization type '{viz_spec.type}'", err=True)
        raise typer.Exit(1)

    output_path = str(output) if output else None
    render_result = plugin.render(exec_result.data, viz_spec, output_path)

    if not render_result.success:
        typer.echo(f"Error rendering visualization: {render_result.error_message}", err=True)
        raise typer.Exit(1)

    if output_path:
        typer.echo(f"\nVisualization saved to: {output_path}")
    else:
        typer.echo(f"\nVisualization rendered ({render_result.data_points_rendered} points)")
        typer.echo("Use --output to save to file.")


@app.command("datasets")
def list_datasets_from_db():
    """List all datasets with extracted metadata from catalog database (primary command)."""
    pg_catalog = get_postgres_catalog()
    if pg_catalog is None:
        typer.echo("Error: DATABASE_URL environment variable not set.", err=True)
        typer.echo("Set it to connect to the catalog database:")
        typer.echo("  export DATABASE_URL='postgresql://user:pass@localhost:5432/gatekeeper_db'")
        typer.echo("\nFor standalone mode in standalone mode, use 'nl2vis-bench list' instead.")
        raise typer.Exit(1)

    files = pg_catalog.list_files_with_metadata()

    if not files:
        typer.echo("No files with extracted metadata found.")
        typer.echo("Run metadata-extractor to process files first.")
        return

    typer.echo(f"Found {len(files)} file(s) with metadata:\n")

    for f in files:
        typer.echo(f"  {f.name}")
        typer.echo(f"    ID: {f.data_file_id}")
        typer.echo(f"    Rows: {f.row_count:,}")
        typer.echo(f"    Columns: {f.column_count}")
        typer.echo(f"    Sample: {'Yes' if f.sample_file_path else 'No'}")
        typer.echo(f"    Extracted: {f.extracted_at}")
        typer.echo()


@app.command("ask")
def ask_dataset(
    file_id: Annotated[str, typer.Argument(help="Data file UUID from the catalog")],
    question: Annotated[str, typer.Argument(help="Natural language question")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output path for visualization")] = None,
    no_viz: Annotated[bool, typer.Option("--no-viz", help="Skip visualization, show data only")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output")] = False,
):
    """Query a catalog dataset using natural language (primary command).

    Uses metadata from PostgreSQL database and sample data from MinIO.
    """
    pg_catalog = get_postgres_catalog()
    if pg_catalog is None:
        typer.echo("Error: DATABASE_URL environment variable not set.", err=True)
        typer.echo("\nFor standalone mode in standalone mode, use 'nl2vis-bench query' instead.")
        raise typer.Exit(1)

    minio = get_minio_gateway()
    config = get_config()

    # Parse UUID
    try:
        data_file_uuid = UUID(file_id)
    except ValueError:
        typer.echo(f"Error: Invalid UUID '{file_id}'", err=True)
        raise typer.Exit(1)

    # Get dataset schema from PostgreSQL (includes LLM descriptions)
    schema = pg_catalog.get_dataset_schema(data_file_uuid)
    if schema is None:
        typer.echo(f"Error: No metadata found for file '{file_id}'", err=True)
        typer.echo("Use 'nl2vis-bench datasets' to see available files.")
        raise typer.Exit(1)

    typer.echo(f"Dataset: {schema.name}")
    typer.echo(f"Columns: {len(schema.columns)}")

    if verbose:
        typer.echo("\nColumn metadata (from database):")
        for col in schema.columns:
            desc = f" - {col.description}" if col.description else ""
            typer.echo(f"  • {col.name} ({col.dtype}){desc}")

    # Load sample data from MinIO
    df = None
    if schema.sample_file_path and minio:
        typer.echo(f"\nLoading sample data from MinIO...")
        try:
            df = minio.get_sample_data(schema.sample_file_path)
            typer.echo(f"Loaded {len(df)} sample rows")
        except Exception as e:
            typer.echo(f"Warning: Could not load sample data: {e}", err=True)
            typer.echo("Continuing with schema-only translation...")
    elif not minio:
        typer.echo("\nNote: MinIO not configured. Set MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY.")
        typer.echo("Continuing with schema-only translation...")
    elif not schema.sample_file_path:
        typer.echo("\nNote: No sample data available for this file.")
        typer.echo("Run metadata-extractor to generate sample data.")

    # Translate question to query
    typer.echo("\nTranslating question...")
    llm = get_llm_provider(config)
    translator = NLTranslator(llm_provider=llm)

    try:
        result = translator.translate(question, schema)
    except ValueError as e:
        typer.echo(f"Error translating question: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nGenerated query: {result.query}")
    typer.echo(f"Columns used: {result.columns_used}")
    typer.echo(f"Explanation: {result.explanation}")

    if verbose:
        typer.echo(f"Latency: {result.latency_ms:.0f}ms")

    # If no sample data, stop here
    if df is None:
        typer.echo("\nQuery generated but cannot execute without sample data.")
        return

    # Execute query
    typer.echo("\nExecuting query...")
    executor = QueryExecutor()
    exec_result = executor.execute(df, result.query)

    if not exec_result.success:
        typer.echo(f"Error executing query: {exec_result.error_message}", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Execution time: {exec_result.execution_time_ms:.0f}ms")
        typer.echo(f"Result rows: {exec_result.profile.row_count}")

    # Show data preview
    typer.echo("\nResult:")
    typer.echo(exec_result.data.to_string(max_rows=10))

    if no_viz:
        return

    # Select visualization
    typer.echo("\nSelecting visualization...")
    selector = VizSelector(llm_provider=llm)
    viz_spec = selector.select(exec_result.profile, question)

    # Enrich with user-friendly labels from column descriptions
    viz_spec = _enrich_viz_labels(viz_spec, schema, question, result.explanation)

    if verbose:
        typer.echo(f"Visualization type: {viz_spec.type}")
        typer.echo(f"X: {viz_spec.x}, Y: {viz_spec.y}")
        typer.echo(f"Title: {viz_spec.title}")
        typer.echo(f"Selected by: {viz_spec.selected_by}")

    # Render visualization
    plugin_registry = PluginRegistry.default()
    plugin = plugin_registry.get(viz_spec.type)

    if plugin is None:
        typer.echo(f"Error: No plugin for visualization type '{viz_spec.type}'", err=True)
        raise typer.Exit(1)

    output_path = str(output) if output else None
    render_result = plugin.render(exec_result.data, viz_spec, output_path)

    if not render_result.success:
        typer.echo(f"Error rendering visualization: {render_result.error_message}", err=True)
        raise typer.Exit(1)

    if output_path:
        typer.echo(f"\nVisualization saved to: {output_path}")
    else:
        typer.echo(f"\nVisualization rendered ({render_result.data_points_rendered} points)")
        typer.echo("Use --output to save to file.")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
