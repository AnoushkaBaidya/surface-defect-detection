PYTHON ?= python

.PHONY: reproduce reproduce-from-saved test lint format-check

reproduce:
	$(PYTHON) -m scripts.reproduce_retained_sample_pipeline

reproduce-from-saved:
	$(PYTHON) -m scripts.reproduce_retained_sample_pipeline --from-saved-predictions

test:
	pytest

lint:
	ruff check .

format-check:
	black --check .
