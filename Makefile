PACKAGE=compotime
TEST_DIR=./tests
DOCS_DIR=./docs

.PHONY: tests lint jupyter format docs

tests:
	pytest --cov=$(PACKAGE) --cov-report=xml -n auto

lint:
	ruff compotime tests

jupyter:
	export PYTHONPATH=$(shell pwd) && jupyter-lab

format:
	black .
	ruff -s --fix --exit-zero .
	isort --profile black -l 100 .

docs: ## Build documentation with Sphinx
	rm -rf docs/source/_autosummary
	rm -rf docs/build/*
	$(MAKE) -C $(DOCS_DIR) html
	
