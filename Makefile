.PHONY: python-install python-run cli test lint

python-install:
	pip install -e ".[dev]"

python-run:
	set -a && source $(ENV_FILE_PATH) && set +a && nl2vis-bench

cli:
	set -a && source $(ENV_FILE_PATH) && set +a && nl2vis-bench $(ARGS)

test:
	pytest -v

lint:
	ruff check nl2vis_bench/
	ruff format nl2vis_bench/
