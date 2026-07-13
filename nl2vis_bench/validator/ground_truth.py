"""Ground truth comparison utilities."""
import re
import yaml
from pathlib import Path
from dataclasses import dataclass


# Module-level cache for equivalence pairs
_equivalence_pairs: dict[str, set[str]] | None = None


@dataclass
class QueryMatch:
    """Result of comparing generated query to ground truth."""
    matches: bool
    generated_normalized: str
    expected_normalized: str
    needs_manual_review: bool = False
    semantic_equivalence_used: bool = False


def load_semantic_equivalence(config_path: Path | None = None) -> dict[str, set[str]]:
    """Load semantic equivalence pairs from YAML config.

    Returns dict mapping each column to its equivalent columns (including itself).
    """
    global _equivalence_pairs
    if _equivalence_pairs is not None:
        return _equivalence_pairs

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "data" / "evaluation" / "semantic_equivalence.yaml"

    if not config_path.exists():
        _equivalence_pairs = {}
        return _equivalence_pairs

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _equivalence_pairs = {}
    for pair in config.get("equivalent_pairs", []):
        cols = pair["columns"]
        col_set = set(cols)
        for col in cols:
            _equivalence_pairs[col] = col_set

    return _equivalence_pairs


def are_columns_equivalent(col1: str, col2: str) -> bool:
    """Check if two columns are semantically equivalent."""
    if col1 == col2:
        return True
    equivalences = load_semantic_equivalence()
    if col1 in equivalences:
        return col2 in equivalences[col1]
    return False


def normalize_query(query: str) -> str:
    """Normalize a Pandas query for comparison.

    Normalization steps:
    1. Remove all whitespace
    2. Convert to lowercase
    3. Standardize quotes to single quotes
    4. Sort column lists alphabetically
    """
    # Remove whitespace
    normalized = re.sub(r'\s+', '', query)

    # Lowercase
    normalized = normalized.lower()

    # Standardize quotes to single quotes
    normalized = normalized.replace('"', "'")

    # Sort column lists: find [['a', 'b']] patterns and sort
    def sort_columns(match):
        cols_str = match.group(1)
        # Extract column names
        cols = re.findall(r"'([^']+)'", cols_str)
        cols_sorted = sorted(cols)
        return "[['" + "','".join(cols_sorted) + "']]"

    normalized = re.sub(r"\[\[([^\]]+)\]\]", sort_columns, normalized)

    return normalized


def normalize_query_with_equivalence(query: str, equivalences: dict[str, set[str]]) -> list[str]:
    """Return list of normalized query variants using equivalent columns.

    For each equivalent pair, generates a query variant with the substitution.
    """
    variants = [normalize_query(query)]

    for col, equiv_set in equivalences.items():
        if col.lower() in query.lower():
            for equiv_col in equiv_set:
                if equiv_col != col:
                    # Create variant with substitution
                    variant = re.sub(
                        rf"['\"]?{re.escape(col)}['\"]?",
                        f"'{equiv_col}'",
                        query,
                        flags=re.IGNORECASE
                    )
                    variants.append(normalize_query(variant))

    return list(set(variants))


def compare_queries(generated: str, expected: str, use_equivalence: bool = True) -> QueryMatch:
    """Compare generated query against expected ground truth.

    Args:
        generated: Generated pandas query
        expected: Expected ground truth query
        use_equivalence: If True, consider semantically equivalent columns as matches

    Returns:
        QueryMatch with match status and normalized forms
    """
    gen_norm = normalize_query(generated)
    exp_norm = normalize_query(expected)

    # Direct match
    if gen_norm == exp_norm:
        return QueryMatch(
            matches=True,
            generated_normalized=gen_norm,
            expected_normalized=exp_norm,
            needs_manual_review=False,
            semantic_equivalence_used=False,
        )

    # Check with semantic equivalence
    if use_equivalence:
        equivalences = load_semantic_equivalence()
        gen_variants = normalize_query_with_equivalence(generated, equivalences)
        exp_variants = normalize_query_with_equivalence(expected, equivalences)

        # Check if any variant pair matches
        for gen_var in gen_variants:
            for exp_var in exp_variants:
                if gen_var == exp_var:
                    return QueryMatch(
                        matches=True,
                        generated_normalized=gen_norm,
                        expected_normalized=exp_norm,
                        needs_manual_review=False,
                        semantic_equivalence_used=True,
                    )

    return QueryMatch(
        matches=False,
        generated_normalized=gen_norm,
        expected_normalized=exp_norm,
        needs_manual_review=False,
        semantic_equivalence_used=False,
    )
