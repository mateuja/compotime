PACKAGE=compotime
TEST_DIR=tests

.PHONY: tests lint format

tests:
	pytest $(TEST_DIR)

lint:
	ruff --fix compotime tests

format:
	black .
	ruff -s --fix --exit-zero .
	isort --profile black -l 100 .
