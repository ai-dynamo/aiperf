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
		test coverage clean install docker docker-run first-time-setup \
		test-verbose init-files setup-venv setup-mkinit \
		test-integration test-integration-verbose \
		internal-help help


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

# Try and get the app name and version from uv
APP_NAME := $(shell $(activate_venv) 2>/dev/null && uv version 2>/dev/null | cut -d ' ' -f 1)
APP_VERSION := $(shell $(activate_venv) 2>/dev/null && uv version 2>/dev/null | cut -d ' ' -f 2)

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
	@$(MAKE) internal-help --no-print-directory | less

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

init-files: #? run mkinit to generate the __init__.py files.
	$(activate_venv) && tools/generate_init_files.sh

ruff lint: #? run the ruff linters
	$(activate_venv) && ruff check . $(args)

ruff-fix lint-fix: #? auto-fix the linter errors of the project using ruff.
	$(activate_venv) && ruff check . --fix $(args)

format fmt: #? format the project using ruff.
	$(activate_venv) && ruff format . $(args)

check-format check-fmt: #? check the formatting of the project using ruff.
	$(activate_venv) && ruff format . --check $(args)

test: #? run the tests using pytest-xdist.
	$(activate_venv) && pytest -n auto -m 'not integration and not performance' $(args)

test-verbose: #? run the tests using pytest-xdist with DEBUG logging.
	$(activate_venv) && pytest -n auto -v -s --log-cli-level -m 'not integration and not performance' DEBUG

coverage: #? run the tests and generate an html coverage report.
	$(activate_venv) && pytest -n auto --cov=aiperf --cov-branch --cov-report=html --cov-report=xml --cov-report=term -m 'not integration and not performance' $(args)

install: #? install the project in editable mode.
	$(activate_venv) && uv pip install -e ".[dev]" $(args)

docker: #? build the docker image.
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) $(args) .

docker-run: #? run the docker container.
	docker run -it --rm $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) $(args)

version: #? print the version of the project.
	@PATH=$(UV_PATH):$(PATH) uv version

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

setup-mkinit: #? install the mkinit and ruff packages for pre-commit.
	$(activate_venv) && uv pip install mkinit ruff

first-time-setup: #? convenience command to setup the environment for the first time
	$(MAKE) setup-venv --no-print-directory

	@# Install the project
	@printf "$(bold)$(green)Installing project...$(reset)\n"
	@PATH=$(UV_PATH):$(PATH) $(MAKE) --no-print-directory install

	@# Install the mock server
	@printf "$(bold)$(green)Installing mock server...$(reset)\n"
	@$(MAKE) -C integration-tests --no-print-directory install

	@# Install pre-commit hooks
	@printf "$(bold)$(green)Installing pre-commit hooks...$(reset)\n"
	$(activate_venv) && pre-commit install --install-hooks

	@# Print a success message
	@printf "$(bold)$(green)Done!$(reset)\n"

test-integration: #? run integration tests with with FakeAI server.
	@printf "$(bold)$(blue)Running integration tests with FakeAI server...$(reset)\n"
	$(activate_venv) && pytest tests/integration/ -n auto -v $(args)
	@printf "$(bold)$(green)Integration tests passed!$(reset)\n"

test-integration-verbose: #? run integration tests with verbose output with FakeAI server.
	@printf "$(bold)$(blue)Running integration tests (verbose, sequential) with FakeAI server...$(reset)\n"
	@printf "$(yellow)Note: Sequential mode shows real-time AIPerf output$(reset)\n"
	$(activate_venv) && pytest tests/integration/ -v -s --log-cli-level=INFO $(args)
	@printf "$(bold)$(green)Integration tests passed!$(reset)\n"
