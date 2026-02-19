#!/bin/bash

set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

function install {
    echo "Installing production dependencies..."
    pip install -e .
}

function install_dev {
    echo "Installing development dependencies..."
    pip install -e ".[dev]"
    pre-commit install
}

function lint {
    echo "Running linting checks..."
    ruff check .
}

function format {
    echo "Formatting code..."
    ruff format .
    ruff check --fix .
}

function type_check {
    echo "Running type checking..."
    mypy .
}

function test {
    echo "Running tests..."
    pytest -v
}

function test_cov {
    echo "Running tests with coverage..."
    pytest -v --cov --cov-report=html --cov-report=term
}

function clean {
    echo "Cleaning generated files and caches..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    rm -rf htmlcov/ .coverage coverage.xml
    rm -rf dist/ build/
}

function help {
    echo "$0 <task> <args>"
    echo "Tasks:"
    compgen -A function | cat -n
}

TIMEFORMAT="Task completed in %3lR"
time ${@:-help}
