# SHELL := /bin/bash

.PHONY: default
default: help


.PHONY: init
init:  ## Run this to initialize the project. Installs poetry, fixes prompt and path, and installs dependencies
	@pip install poetry==1.7.0; \
	make install;

.PHONY: install
install:
	poetry config virtualenvs.in-project true; \
	poetry install; \
	poetry run pre-commit install;

.PHONY: bump
bump:  ## Bump the patch version of the project
	@poetry run bump-my-version bump patch

.PHONY: coverage
coverage:  ## Run pytest with coverage
	@poetry run pytest --cov=src --cov-report=term-missing --cov-report=xml tests/

.PHONY: test
test:  ## Run unit tests
	@poetry run pytest

.PHONY: help
help:
	@echo "Usage: make <target>"; \
	grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
