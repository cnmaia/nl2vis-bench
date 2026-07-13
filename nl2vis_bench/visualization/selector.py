"""Visualization type selection using heuristics and LLM fallback."""
import json
from nl2vis_bench.models import DataProfile, VizSpec
from nl2vis_bench.llm.base import LLMProvider


SELECTOR_PROMPT = """You are a data visualization expert. Given the data profile below, select the best visualization type.

DATA PROFILE:
- Row count: {row_count}
- Has datetime column: {has_datetime} ({datetime_col})
- Numeric columns ({numeric_count}): {numeric_cols}
- Categorical columns: {categorical_cols}

USER QUESTION: "{question}"

Choose ONE visualization type from: line, histogram, scatter

Respond ONLY with valid JSON:
{{"type": "line|histogram|scatter", "x": "column_name", "y": "column_name or null", "reasoning": "brief explanation"}}"""


class VizSelector:
    """Selects visualization type based on data profile."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm = llm_provider

    def select(
        self,
        profile: DataProfile,
        question: str = "",
        use_llm_fallback: bool = True,
    ) -> VizSpec:
        """Select best visualization type for the data."""
        # Try heuristics first
        spec = self._select_by_heuristics(profile, question)

        if spec is not None:
            return spec

        # Fallback to LLM if available and enabled
        if use_llm_fallback and self.llm is not None:
            return self._select_by_llm(profile, question)

        # Default to histogram if nothing else matches
        x_col = profile.numeric_columns[0] if profile.numeric_columns else "result"
        return VizSpec(
            type="histogram",
            x=x_col,
            selected_by="heuristic",
            reasoning="Default fallback to histogram",
        )

    def _select_by_heuristics(self, profile: DataProfile, question: str) -> VizSpec | None:
        """Apply heuristic rules for visualization selection."""
        question_lower = question.lower()

        # Rule 1: Time series data with datetime → line chart
        if profile.has_datetime_column and profile.has_numeric_columns:
            # Check for time-related keywords
            time_keywords = ["over time", "trend", "evolution", "temporal", "series", "ao longo"]
            if any(kw in question_lower for kw in time_keywords) or profile.row_count > 10:
                y_col = profile.numeric_columns[0]
                return VizSpec(
                    type="line",
                    x=profile.datetime_column,
                    y=y_col,
                    selected_by="heuristic",
                    reasoning="Time series data detected",
                )

        # Rule 2: Distribution keywords → histogram
        dist_keywords = ["distribution", "histogram", "frequency", "distribuicao", "frequencia"]
        if any(kw in question_lower for kw in dist_keywords):
            if profile.numeric_columns:
                return VizSpec(
                    type="histogram",
                    x=profile.numeric_columns[0],
                    selected_by="heuristic",
                    reasoning="Distribution analysis requested",
                )

        # Rule 3: Correlation/relationship keywords → scatter
        corr_keywords = ["correlation", "relationship", "versus", "vs", "scatter", "correlacao", "relacao"]
        if any(kw in question_lower for kw in corr_keywords):
            if profile.numeric_column_count >= 2:
                return VizSpec(
                    type="scatter",
                    x=profile.numeric_columns[0],
                    y=profile.numeric_columns[1],
                    selected_by="heuristic",
                    reasoning="Correlation analysis requested",
                )

        # Rule 4: Single numeric column → histogram
        if profile.numeric_column_count == 1 and not profile.has_datetime_column:
            return VizSpec(
                type="histogram",
                x=profile.numeric_columns[0],
                selected_by="heuristic",
                reasoning="Single numeric column",
            )

        # Rule 5: Two numeric columns, no datetime → scatter
        if profile.numeric_column_count == 2 and not profile.has_datetime_column:
            return VizSpec(
                type="scatter",
                x=profile.numeric_columns[0],
                y=profile.numeric_columns[1],
                selected_by="heuristic",
                reasoning="Two numeric columns without time",
            )

        # Rule 6: Datetime + numeric → default to line
        if profile.has_datetime_column and profile.has_numeric_columns:
            return VizSpec(
                type="line",
                x=profile.datetime_column,
                y=profile.numeric_columns[0],
                selected_by="heuristic",
                reasoning="Time series data",
            )

        return None

    def _select_by_llm(self, profile: DataProfile, question: str) -> VizSpec:
        """Use LLM to select visualization type."""
        prompt = SELECTOR_PROMPT.format(
            row_count=profile.row_count,
            has_datetime=profile.has_datetime_column,
            datetime_col=profile.datetime_column or "none",
            numeric_count=profile.numeric_column_count,
            numeric_cols=", ".join(profile.numeric_columns),
            categorical_cols=", ".join(profile.categorical_columns),
            question=question,
        )

        response = self.llm.complete(prompt)

        try:
            # Parse LLM response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            data = json.loads(response.strip())

            return VizSpec(
                type=data["type"],
                x=data["x"],
                y=data.get("y"),
                selected_by="llm",
                reasoning=data.get("reasoning", "LLM selection"),
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback to default
            x_col = profile.numeric_columns[0] if profile.numeric_columns else "result"
            return VizSpec(
                type="histogram",
                x=x_col,
                selected_by="llm",
                reasoning="LLM response parsing failed, using default",
            )
