# Copyright (c) 2022-2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Bulldozer
# (see https://github.com/CNES/bulldozer).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Initially based on Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# Dependencies : python3 venv
# Some Makefile global variables can be set in make command line
# Recall: .PHONY  defines special targets not associated with files

# Bulldozer Makefile
# 
# Unified Makefile for contributors and maintainers.
#
# - Local development
# - Code quality & tests
# - Documentation

.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# -------------------------------------------------------------------
# Configuration (overridable)
# -------------------------------------------------------------------

VENV ?= bulldozer_venv
PYTHON ?= python3
PYTHON_VERSION_MIN = 3.10

PYTHON_CMD := $(shell command -v $(PYTHON))
ifeq ($(PYTHON_CMD),)
  $(error "Python executable '$(PYTHON)' not found in PATH")
endif

PYTHON_VERSION_CUR := $(shell $(PYTHON_CMD) -c 'import sys; print("%d.%d"%sys.version_info[:2])')
PYTHON_VERSION_OK := $(shell $(PYTHON_CMD) -c 'import sys; print(int(sys.version_info[:2] >= tuple(map(int,"$(PYTHON_VERSION_MIN)".split(".")))))')

ifeq ($(PYTHON_VERSION_OK),0)
  $(error "Requires Python >= $(PYTHON_VERSION_MIN), current version is $(PYTHON_VERSION_CUR)")
endif

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@echo "Bulldozer Makefile"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Common workflow:"
	@echo "  make install"
	@echo "  make check"
	
# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

.PHONY: venv
venv: ## Create virtual environment
venv:
	@test -d $(VENV) || (echo "Creating virtualenv in $(VENV)"; $(PYTHON_CMD) -m venv $(VENV))
	@$(VENV)/bin/python -m pip install --upgrade pip setuptools wheel

.PHONY: install
install: venv ## Install Bulldozer in editable mode with dev dependencies
	@$(VENV)/bin/pip install -e .[dev,docs]
	@$(VENV)/bin/pre-commit install --install-hooks
	@$(VENV)/bin/pre-commit install --hook-type pre-commit
	@echo ""
	@echo "Bulldozer installed in virtualenv ${VENV}."
	@echo "Activate with: source $(VENV)/bin/activate"

# -------------------------------------------------------------------
# Code formatting
# -------------------------------------------------------------------

.PHONY: format
format: install ## Format code with isort and black
	@$(VENV)/bin/isort bulldozer tests
	@$(VENV)/bin/black bulldozer tests

# -------------------------------------------------------------------
# Linting & code quality
# -------------------------------------------------------------------

.PHONY: lint
lint: install ## Run all linters via pre-commit
	@$(VENV)/bin/pre-commit run --all-files

.PHONY: lint-manual
lint-manual: install ## Run linters individually (debug)
	@$(VENV)/bin/isort --check bulldozer tests
	@$(VENV)/bin/black --check bulldozer tests
	@$(VENV)/bin/flake8 bulldozer tests
	@$(VENV)/bin/mypy bulldozer
	@set -o pipefail; \
	  $(VENV)/bin/pylint bulldozer tests --rcfile=.pylintrc | tee pylint-report.txt

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

.PHONY: test
test: install ## Run test
	@$(VENV)/bin/pytest -o log_cli=true

.PHONY: coverage
coverage: install ## Run tests with coverage
	@$(VENV)/bin/pytest --cov --cov-config=.coveragerc
	@$(VENV)/bin/coverage report -m
	@$(VENV)/bin/coverage html

.PHONY: check
check: lint test ## Run all checks (CI equivalent)

.PHONY: ci
ci: check ## Alias for CI pipeline

# -------------------------------------------------------------------
# Documentation
# -------------------------------------------------------------------

.PHONY: docs
docs: install ## generate Mkdocs HTML documentation
	@$(VENV)/bin/mkdocs build --clean --strict

# -------------------------------------------------------------------
# Cleaning
# -------------------------------------------------------------------

.PHONY: clean
clean: ## Remove virtualenv and temporary files
	@rm -rf $(VENV)
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .pytest_cache .mypy_cache .coverage htmlcov site
	@rm -f pylint-report.txt
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.so" -delete
	@find . -type f -name "*.cpp" ! -name "c_*.cpp" -delete
