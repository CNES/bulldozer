# Copyright 2021 PIERRE LASSALLE
# All rights reserved

# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

.PHONY: help test lint

help:  ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:  ## install environment for development target
	@virtualenv -p `which python3` .venv
	@.venv/bin/pip install -e .[dev]
	@.venv/bin/pre-commit install -t pre-commit
	@.venv/bin/pre-commit install -t pre-push

test:  ## run tests (requires venv activation)
	@pytest --show-capture=no --cov-report html --cov-report term-missing --cov --cov-fail-under=80

lint:  ## run black and isort (requires venv activation)
	@isort **/*.py
	@black **/*.py

git:  ## create initial commit
	@git init
	@git add .
	@git commit -am 'Initial commit'
