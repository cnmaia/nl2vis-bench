"""Query validation for translated queries."""
from dataclasses import dataclass
from nl2vis_bench.models import DatasetSchema


ALLOWED_OPERATIONS = {
    'groupby', 'mean', 'sum', 'count', 'min', 'max', 'std', 'var', 'median',
    'filter', 'query', 'loc', 'iloc', 'head', 'tail',
    'sort_values', 'sort_index',
    'resample', 'rolling', 'dt', 'str',
    'dropna', 'fillna', 'astype', 'reset_index',
    '__getitem__',
}

FORBIDDEN_PATTERNS = ['import', 'exec', 'eval', 'open', 'os.', 'subprocess', '__']


@dataclass
class ValidationResult:
    """Result of query validation."""
    is_valid: bool
    error: str | None = None


class QueryValidator:
    """Validates translated queries against schema and security rules."""

    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.valid_columns = {col.name for col in schema.columns}

    def validate_columns(self, columns: list[str]) -> ValidationResult:
        """Check that all columns exist in the schema."""
        invalid = [c for c in columns if c not in self.valid_columns]
        if invalid:
            return ValidationResult(
                is_valid=False,
                error=f"Invalid columns: {invalid}. Valid columns are: {list(self.valid_columns)}"
            )
        return ValidationResult(is_valid=True)

    def validate_syntax(self, query: str) -> ValidationResult:
        """Check that query is syntactically valid Python."""
        try:
            compile(query, '<string>', 'eval')
            return ValidationResult(is_valid=True)
        except SyntaxError as e:
            return ValidationResult(is_valid=False, error=f"Syntax error: {e}")

    def validate_security(self, query: str) -> ValidationResult:
        """Check for forbidden patterns."""
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in query:
                return ValidationResult(
                    is_valid=False,
                    error=f"Forbidden pattern detected: {pattern}"
                )
        return ValidationResult(is_valid=True)

    def validate(self, query: str, columns: list[str]) -> ValidationResult:
        """Run all validations."""
        for check in [
            lambda: self.validate_columns(columns),
            lambda: self.validate_syntax(query),
            lambda: self.validate_security(query),
        ]:
            result = check()
            if not result.is_valid:
                return result
        return ValidationResult(is_valid=True)
