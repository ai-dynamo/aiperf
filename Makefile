# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# This is a command-line tool Makefile for the AIPerf project.
#
# It is being used to support common development workflow commands without
# having to remember all the the specific flags for each one. Everything
# done in here can be done manually but this is just a convenience.
#
# *** NOTICE: ***
# Commands here are not guaranteed to work with every possible configuration
# of the development environment, or to even work at all. Users are encouraged
# to read the source code and documentation for more information on how to use
# the project.


.PHONY: ruff lint ruff-fix lint-fix format fmt check-format check-fmt \
		test coverage clean install install-app docker docker-run first-time-setup \
		test-verbose setup-venv install-mock-server test-ci test-all \
		integration-tests integration-tests-ci integration-tests-verbose integration-tests-ci-macos \
		test-integration test-integration-ci test-integration-verbose test-integration-ci-macos \
		test-component-integration test-component-integration-ci test-component-integration-verbose \
		add-copyright generate-cli-docs generate-env-vars-docs generate-plugin-enums \
		generate-plugin-overloads check-plugin-overloads generate-plugin-schemas \
		generate-all-plugin-files generate-all-docs test-stress stress-tests internal-help help


# Include user-defined environment variables
-include .env.mk

SHELL := /bin/bash

PROJECT_NAME ?= AIPerf

# The path to the virtual environment
VENV_PATH ?= .venv
# The python version to use
PYTHON_VERSION ?= 3.12
# The command to activate the virtual environment
activate_venv = . $(VENV_PATH)/bin/activate

# Try and get the app name and version from pyproject.toml
APP_NAME := $(shell grep '^name = ' pyproject.toml 2>/dev/null | sed 's/name = "\(.*\)"/\1/')
APP_VERSION := $(shell grep '^version = ' pyproject.toml 2>/dev/null | sed 's/version = "\(.*\)"/\1/')

# The folder where uv is installed
UV_PATH ?= $(HOME)/.local/bin

# The name of the docker image (defaults to the app name)
DOCKER_IMAGE_NAME ?= $(APP_NAME)
# The tag of the docker image (defaults to the app version)
DOCKER_IMAGE_TAG ?= $(APP_VERSION)

# The extra arguments the user passed to make
args = $(filter-out $@,$(MAKECMDGOALS))

# Color and style definitions
red := $(shell tput setaf 1)
green := $(shell tput setaf 2)
yellow := $(shell tput setaf 3)
blue := $(shell tput setaf 4)
reset := $(shell tput sgr0)
bold := $(shell tput bold)
italic := $(shell tput sitm)
dim := $(shell tput dim)

.DEFAULT_GOAL := help


help: #? show this help
	@$(MAKE) internal-help --no-print-directory

#
# Help command is automatically generated based on the comments in the Makefile.
# Place a comment after each make target in the format `#? <command description>`
# to include it in the help command.
#
# NOTE: Currently the help command does not support more than 1 alias for a single target.
#       any more than one alias will cause the help command to not show the target.
#
# Internal Commands:
# DO NOT add #? documentation regarding this internal-help command
# to avoid it being included in the external facing list of commands.
internal-help:
	@printf "──────────────────────────────$(bold)$(blue) AIPerf Makefile $(reset)──────────────────────────────\n"
	@printf "$(bold)$(italic)$(yellow) NOTICE:$(reset)$(italic) Commands here are not guaranteed to work with every possible$(reset)\n"
	@printf "$(italic) configuration of the development environment, or to even work at all.$(reset)\n"
	@printf "$(italic) Users are encouraged to read the source code and documentation for more$(reset)\n"
	@printf "$(italic) information on how to use the project.$(reset)\n"
	@printf "───────────────────────────────$(bold)$(blue) Make Commands $(reset)───────────────────────────────\n"
	@{ \
		sed -ne "/@sed/!s/^\([^ :]*\)\s\+\([^ :]*\):\s*#?\(.*\)/$(bold)$(green)\1$(reset) $(dim)[\2$(reset)$(dim)]$(reset):$(italic)\3$(reset)/p" $(MAKEFILE_LIST); \
		sed -ne "/@sed/!s/^\([^ :]*\):\s*#?\(.*\)/$(bold)$(green)\1$(reset):$(italic)\2$(reset)/p" $(MAKEFILE_LIST) | grep -v " \["; \
	} | sort
	@printf "────────────────────────────────────────────────────────────────────────────\n"

ruff lint: #? run the ruff linters
	$(activate_venv) && ruff check . $(args)

ruff-fix lint-fix: #? auto-fix the linter errors of the project using ruff.
	$(activate_venv) && ruff check . --fix $(args)

format fmt: #? format the project using ruff.
	$(activate_venv) && ruff format . $(args)

check-format check-fmt: #? check the formatting of the project using ruff.
	$(activate_venv) && ruff format . --check $(args)

test: #? run the tests using pytest-xdist.
	$(activate_venv) && pytest tests/unit -n auto -m 'not integration and not performance and not component_integration' $(args)

test-verbose: #? run the tests using pytest-xdist with DEBUG logging.
	$(activate_venv) && pytest tests/unit -n auto -v -s --log-cli-level=DEBUG -m 'not integration and not performance and not component_integration'

test-imports: #? verify all modules (src and tests) can be imported.
	$(activate_venv) && pytest tests/unit/test_imports.py -q $(args)

test-imports-src: #? verify all modules in src/aiperf can be imported.
	$(activate_venv) && pytest tests/unit/test_imports.py::test_all_aiperf_modules_can_be_imported -q $(args)

test-imports-tests: #? verify all modules in tests/ can be imported.
	$(activate_venv) && pytest tests/unit/test_imports.py::test_all_test_modules_can_be_imported -q $(args)

coverage: #? run the tests and generate an html coverage report.
	$(activate_venv) && pytest tests/unit -n auto --cov=src/aiperf --cov-branch --cov-report=html --cov-report=xml --cov-report=term -m 'not integration and not performance and not component_integration' $(args)

install: install-app install-mock-server #? install the project and mock server in editable mode.

install-app: #? install the project in editable mode.
	$(activate_venv) && uv pip install -e ".[dev]"

docker: #? build the docker image.
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) $(args) .

docker-run: #? run the docker container.
	docker run -it --rm $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) $(args)

version: #? print the version of the project.
	@PATH=$(UV_PATH):$(PATH) uv version

install-mock-server: #? install the mock server in editable mode.
	$(activate_venv) && uv pip install -e "tests/aiperf_mock_server[dev]"

clean: #? clean up the pytest and ruff caches, coverage reports, and *.pyc files.
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/

setup-venv: #? create the virtual environment.
	@# Install uv if it is not installed
	@export PATH=$(UV_PATH):$(PATH) && \
	if ! command -v uv &> /dev/null; then \
		printf "$(bold)$(green)Installing uv...$(reset)\n"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		printf "$(bold)$(green)uv already installed$(reset)\n"; \
	fi

	@# Create virtual environment if it does not exist
	@export PATH=$(UV_PATH):$(PATH) && \
	if [ ! -d "$(VENV_PATH)" ]; then \
		printf "$(bold)$(green)Creating virtual environment...$(reset)\n"; \
		uv venv --python $(PYTHON_VERSION); \
	else \
		printf "$(bold)$(green)Virtual environment already exists$(reset)\n"; \
	fi

first-time-setup: #? convenience command to setup the environment for the first time
	$(MAKE) setup-venv --no-print-directory

	@# Install the project
	@printf "$(bold)$(green)Installing project...$(reset)\n"
	@PATH=$(UV_PATH):$(PATH) $(MAKE) --no-print-directory install

	@# Install the mock server
	@printf "$(bold)$(green)Installing mock server...$(reset)\n"
	@PATH=$(UV_PATH):$(PATH) $(MAKE) --no-print-directory install-mock-server

	@# Generate plugin enum stubs for IDE autocomplete
	@printf "$(bold)$(green)Generating plugin enum stubs...$(reset)\n"
	@PATH=$(UV_PATH):$(PATH) $(MAKE) --no-print-directory generate-plugin-enums

	@# Generate plugin overloads for IDE autocomplete
	@printf "$(bold)$(green)Generating plugin overloads...$(reset)\n"
	@PATH=$(UV_PATH):$(PATH) $(MAKE) --no-print-directory generate-plugin-overloads

	@# Install pre-commit hooks
	@printf "$(bold)$(green)Installing pre-commit hooks...$(reset)\n"
	$(activate_venv) && pre-commit install --install-hooks

	@# Print a success message
	@printf "$(bold)$(green)Done!$(reset)\n"

test-all: #? run all tests (unit, component integration, and integration).
	make test --no-print-directory
	make test-component-integration --no-print-directory
	make test-integration --no-print-directory

test-ci: #? run the tests using pytest-xdist for CI.
	@printf "$(bold)$(blue)Running unit and component integration tests (CI mode)...$(reset)\n"
	@# Run unit tests first with coverage
	@printf "$(bold)$(blue)Running unit tests...$(reset)\n"
	@$(activate_venv) && pytest tests/unit -n auto --cov=src/aiperf --cov-branch --cov-report= -m 'not performance and not stress' --tb=short $(args) || exit_code=$$?; \
	# Run component integration tests with coverage append regardless of unit test result \
	printf "$(bold)$(blue)Running component integration tests...$(reset)\n"; \
	$(activate_venv) && pytest tests/component_integration -n auto --cov=src/aiperf --cov-branch --cov-append --cov-report=html --cov-report=xml --cov-report=term -m 'not performance and not stress' -v --tb=short $(args) || exit_code=$$((exit_code + $$?)); \
	if [[ $$exit_code -eq 0 ]]; then \
		printf "$(bold)$(green)AIPerf unit and component integration tests (CI mode) passed!$(reset)\n"; \
	else \
		printf "$(bold)$(red)AIPerf tests failed with exit code $$exit_code$(reset)\n"; \
		exit $$exit_code; \
	fi

stress-tests test-stress: #? run stress tests with with AIPerf Mock Server.
	@printf "$(bold)$(blue)Running stress tests with AIPerf Mock Server...$(reset)\n"
	$(activate_venv) && pytest tests/integration/ -m 'integration and stress' -vv -s --tb=short --log-cli-level=INFO --capture=no $(args)
	@printf "$(bold)$(green)AIPerf Mock Server stress tests passed!$(reset)\n"

integration-tests test-integration: #? run integration tests with with AIPerf Mock Server.
	@printf "$(bold)$(blue)Running integration tests with AIPerf Mock Server...$(reset)\n"
	$(activate_venv) && pytest tests/integration/ -m 'integration and not stress and not performance' -n auto --tb=short --no-looptime $(args)
	@printf "$(bold)$(green)AIPerf Mock Server integration tests passed!$(reset)\n"

integration-tests-ci test-integration-ci: #? run integration tests with with AIPerf Mock Server for CI (parallel, verbose, no performance and no ffmpeg tests).
	@printf "$(bold)$(blue)Running integration tests (CI mode) with AIPerf Mock Server...$(reset)\n"
	$(activate_venv) && pytest tests/integration/ -m 'integration and not performance and not ffmpeg and not stress' -n auto -v --tb=long $(args)
	@printf "$(bold)$(green)AIPerf Mock Server integration tests (CI mode) passed!$(reset)\n"

integration-tests-ci-macos test-integration-ci-macos: #? run integration tests with with AIPerf Mock Server for CI on macOS (non-parallel, verbose, no performance and no ffmpeg tests).
	@printf "$(bold)$(blue)Running integration tests (CI mode on macOS) with AIPerf Mock Server...$(reset)\n"
	$(activate_venv) && pytest tests/integration/ -m 'integration and not performance and not ffmpeg and not stress' -v --tb=long $(args)
	@printf "$(bold)$(green)AIPerf Mock Server integration tests (CI mode on macOS) passed!$(reset)\n"

integration-tests-verbose test-integration-verbose: #? run integration tests with verbose output with AIPerf Mock Server.
	@printf "$(bold)$(blue)Running integration tests (verbose, sequential) with AIPerf Mock Server...$(reset)\n"
	@printf "$(yellow)Note: Sequential mode shows real-time AIPerf output$(reset)\n"
	$(activate_venv) && pytest tests/integration/ -m 'integration and not stress and not performance' -vv -s --tb=short --log-cli-level=INFO --capture=no $(args)
	@printf "$(bold)$(green)AIPerf Mock Server integration tests passed!$(reset)\n"

component-integration-tests test-component-integration: #? run component integration tests with with AIPerf Mock Server.
	@printf "$(bold)$(blue)Running Fake Component Integration tests...$(reset)\n"
	$(activate_venv) && pytest tests/component_integration/ -m 'component_integration and not stress and not performance' -n auto --tb=short $(args)
	@printf "$(bold)$(green)AIPerf Fake Component Integration tests passed!$(reset)\n"

component-integration-tests-ci test-component-integration-ci: #? run component integration tests with with AIPerf Mock Server for CI (parallel, verbose, no performance and no ffmpeg tests).
	@printf "$(bold)$(blue)Running Fake Component Integration tests (CI mode)...$(reset)\n"
	$(activate_venv) && pytest tests/component_integration/ -m 'component_integration and not performance and not ffmpeg and not stress' -n auto -v --tb=long $(args)
	@printf "$(bold)$(green)AIPerf Fake Component Integration tests (CI mode) passed!$(reset)\n"

component-integration-tests-verbose test-component-integration-verbose: #? run component integration tests with verbose output with AIPerf Mock Server.
	@printf "$(bold)$(blue)Running Fake Component Integration tests (verbose, sequential)...$(reset)\n"
	@printf "$(yellow)Note: Sequential mode shows real-time AIPerf output$(reset)\n"
	$(activate_venv) && pytest tests/component_integration/ -m 'component_integration and not stress and not performance' -vv -s --tb=short --log-cli-level=INFO --capture=no $(args)
	@printf "$(bold)$(green)AIPerf Fake Component Integration tests passed!$(reset)\n"

generate-cli-docs: #? generate the CLI documentation.
	$(activate_venv) && ./tools/generate_cli_docs.py

generate-env-vars-docs: #? generate the environment variables documentation.
	$(activate_venv) && ./tools/generate_env_vars_docs.py

generate-plugin-enums: #? generate the plugin enum stubs (enums.py and enums.pyi).
	$(activate_venv) && ./tools/generate_plugin_artifacts.py --enums

generate-plugin-overloads: #? generate the get_class() overloads in plugins.py.
	$(activate_venv) && ./tools/generate_plugin_artifacts.py --overloads

check-plugin-overloads: #? check if the get_class() overloads are up-to-date.
	$(activate_venv) && ./tools/generate_plugin_artifacts.py --overloads --check

generate-plugin-schemas: #? generate JSON schemas for categories.yaml and plugins.yaml.
	$(activate_venv) && ./tools/generate_plugin_artifacts.py --schemas

validate-plugin-schemas: #? validate categories.yaml and plugins.yaml against their schemas.
	$(activate_venv) && ./tools/generate_plugin_artifacts.py --validate

generate-all-plugin-files: #? generate all plugin files (enums, overloads, schemas).
	$(activate_venv) && ./tools/generate_plugin_artifacts.py

generate-all-docs: #? generate all documentation files.
	$(activate_venv) && ./tools/generate_cli_docs.py
	$(activate_venv) && ./tools/generate_env_vars_docs.py

add-copyright: #? add the copyright header to the files.
	$(activate_venv) && ./tools/add_copyright.py
