"""Prompt templates for NL translation."""

TRANSLATION_PROMPT_MINIMAL = """You are a data analyst assistant helping create DATA VISUALIZATIONS. Translate the user's question into a pandas query that produces chart-worthy data.

DATASET: {dataset_name}

COLUMNS:
{columns_info}

VISUALIZATION RULES (IMPORTANT):
1. Use ONLY columns listed above
2. The DataFrame variable is always 'df'
3. For dates, use df['column'].dt to extract parts
4. Return only the pandas expression, not complete code
5. Always produce data suitable for charts (multiple rows, not scalar values)
6. For time-series data, prefer df[['time_col', 'value_col']] format

USER QUESTION: "{question}"

Respond in JSON:
{{
  "query": "pandas expression using df",
  "columns_used": ["col1", "col2"],
  "explanation": "brief explanation"
}}
"""

TRANSLATION_PROMPT = """You are a data analyst assistant helping create DATA VISUALIZATIONS. Translate the user's question into a pandas query that produces chart-worthy data.

DATASET: {dataset_name}
DESCRIPTION: {dataset_description}
CATEGORY: {dataset_category}

COLUMNS:
{columns_info}

SAMPLE DATA (first 5 rows):
{sample_data}

VISUALIZATION RULES (IMPORTANT):
1. Use ONLY columns listed above
2. The DataFrame variable is always 'df'
3. For dates, use df['column'].dt to extract parts
4. Return only the pandas expression, not complete code

5. **CRITICAL FOR VISUALIZATION**: Always produce data suitable for charts:
   - If the user asks for an aggregate (average, sum, max, min, count, etc.), GROUP BY a time column (year, month, day) or category column to show trends/patterns
   - NEVER return a single scalar value - always return a DataFrame or Series with multiple data points
   - Prefer time-series data when a time/date column exists
   - Example: "average temperature" → group by month/year to show temperature trend, NOT df['temp'].mean()

6. For aggregation questions, identify the best grouping:
   - If there's a time column: prefer KEEPING the original time resolution for line charts
   - For daily data: use the raw time series (e.g., df[['time', 'value']]) - do NOT aggregate
   - Only aggregate if explicitly asked (e.g., "monthly average" → group by month)
   - If you must aggregate, prefer finer granularity (month > year) for better visualizations
   - Always reset_index() after groupby to get a proper DataFrame

7. For questions like "average X" or "show X over time" on time-series data:
   - Return df[['time', 'column']] to show the raw time series
   - This produces a line chart showing trends over time
   - Example: "average temperature" on daily data → df[['time', 't2m_mean']] (NOT grouped)

USER QUESTION: "{question}"

Respond in JSON:
{{
  "query": "pandas expression using df that returns multiple rows for visualization",
  "columns_used": ["col1", "col2"],
  "explanation": "user-friendly explanation of what was done"
}}
"""


def format_columns_info(columns: list) -> str:
    """Format column information for prompt."""
    lines = []
    for col in columns:
        line = f"- {col.name} ({col.dtype})"
        if col.description:
            line += f": {col.description}"
        if col.min_value is not None:
            line += f"\n  Range: {col.min_value} to {col.max_value}"
            if col.mean_value is not None:
                line += f", Mean: {col.mean_value:.2f}"
        if col.sample_values:
            line += f"\n  Samples: {', '.join(str(v) for v in col.sample_values[:3])}"
        lines.append(line)
    return "\n".join(lines)


def format_columns_info_minimal(columns: list) -> str:
    """Format column information without enrichment (name and type only)."""
    lines = []
    for col in columns:
        lines.append(f"- {col.name} ({col.dtype})")
    return "\n".join(lines)


def build_translation_prompt(question: str, schema) -> str:
    """Build the full translation prompt from a question and dataset schema."""
    columns_info = format_columns_info(schema.columns)

    # Format sample data if available
    sample_data = ""
    if hasattr(schema, 'sample_data') and schema.sample_data:
        sample_data = str(schema.sample_data)
    else:
        sample_data = "Not available"

    return TRANSLATION_PROMPT.format(
        dataset_name=schema.name,
        dataset_description=getattr(schema, 'description', '') or '',
        dataset_category=getattr(schema, 'category', '') or '',
        columns_info=columns_info,
        sample_data=sample_data,
        question=question,
    )


def build_translation_prompt_minimal(question: str, schema) -> str:
    """Build translation prompt with minimal column info (no enrichment)."""
    columns_info = format_columns_info_minimal(schema.columns)

    return TRANSLATION_PROMPT_MINIMAL.format(
        dataset_name=schema.name,
        columns_info=columns_info,
        question=question,
    )
