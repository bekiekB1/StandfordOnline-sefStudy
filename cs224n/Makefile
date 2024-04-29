# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	flake8 .
	python3 -m isort .
	pyupgrade

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	find . -name '.pytest_cache' -exec rm -fr {} +

clean: clean-pyc clean-test style
	find . -name '.my_cache' -exec rm -fr {} +
	rm -rf logs/

test: clean
	pytest --cov=emply_attr --cov-report=term-missing --cov-fail-under 90

clear-cell-nbs:
	python3 clear_cell_nbs.py

# Pre-commit
.PHONY: pre-commit-setup
pre-commit-setup: clear-cell-nbs
	@if [ ! -f .pre-commit-config.yaml ]; then \
		pre-commit sample-config > .pre-commit-config.yaml; \
		echo "Generated .pre-commit-config.yaml"; \
	else \
		echo ".pre-commit-config.yaml already exists"; \
	fi
	pre-commit install
	pre-commit autoupdate

configure:
	@echo "Copying configuration files..."
	cp config/project.toml project.toml
