"""NL to Query translator using LLM."""
import json
import time
from nl2vis_bench.models import DatasetSchema, QueryResult
from nl2vis_bench.llm.base import LLMProvider
from .prompts import build_translation_prompt, build_translation_prompt_minimal
from .validator import QueryValidator


class NLTranslator:
    """Translates natural language questions to Pandas queries."""

    def __init__(self, llm_provider: LLMProvider, max_retries: int = 2):
        self.llm = llm_provider
        self.max_retries = max_retries

    def translate(self, question: str, schema: DatasetSchema, use_enrichment: bool = True) -> QueryResult:
        """Translate a natural language question to a Pandas query.

        Args:
            question: Natural language question
            schema: Dataset schema with column information
            use_enrichment: If True, use enriched prompts with descriptions/stats.
                          If False, use minimal prompts with only column names/types.
        """
        if use_enrichment:
            prompt = build_translation_prompt(question, schema)
        else:
            prompt = build_translation_prompt_minimal(question, schema)
        validator = QueryValidator(schema)

        start_time = time.time()
        raw_response = self.llm.complete(prompt)
        latency_ms = (time.time() - start_time) * 1000

        parsed = self._parse_response(raw_response)

        # Validate
        validation = validator.validate(
            parsed.get("query", ""),
            parsed.get("columns_used", [])
        )

        if not validation.is_valid:
            # Could retry here with error context
            raise ValueError(f"Query validation failed: {validation.error}")

        return QueryResult(
            query=parsed["query"],
            columns_used=parsed["columns_used"],
            explanation=parsed.get("explanation", ""),
            llm_provider=self.llm.model_name(),
            latency_ms=latency_ms,
            raw_response=raw_response,
        )

    def _parse_response(self, response: str) -> dict:
        """Parse JSON response from LLM."""
        # Clean up common issues
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        return json.loads(response.strip())
