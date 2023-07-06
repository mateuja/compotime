PACKAGE=compotime
TEST_DIR=./tests
DOCS_DIR=./docs

.PHONY: tests lint format docs

tests:
	pytest $(TEST_DIR)

lint:
	ruff compotime tests

format:
	black .
	ruff -s --fix --exit-zero .
	isort --profile black -l 100 .

docs: ## Build documentation with Sphinx
	$(MAKE) -C $(DOCS_DIR) html
	
