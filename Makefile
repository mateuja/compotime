PACKAGE=compotime
TEST_DIR=./tests
DOCS_DIR=./docs

.PHONY: tests cov lint format docs

tests:
	pytest $(TEST_DIR)

lint:
	ruff compotime tests

format:
	black .
	ruff -s --fix --exit-zero .
	isort --profile black -l 100 .

docs: ## Build documentation with Sphinx
	rm -rf docs/source/_autosummary
	rm -rf docs/build/*
	$(MAKE) -C $(DOCS_DIR) html
	
