# Face Pipeline Makefile
# Development and testing commands for the face processing pipeline

.PHONY: help install install-dev run test format lint clean docker-build docker-run

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Face Pipeline - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Installation
# ============================================================================

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies (includes production deps)
	pip install -r requirements.txt

# ============================================================================
# Development
# ============================================================================

run: ## Run the FastAPI server with auto-reload
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the FastAPI server in production mode
	uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests with pytest
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=pipeline --cov-report=html --cov-report=term

test-unit: ## Run unit tests only (skip integration tests)
	pytest tests/ -v -m "not integration"

test-integration: ## Run integration tests only
	pytest tests/ -v -m integration

test-watch: ## Run tests in watch mode (requires pytest-watch)
	pytest-watch tests/ -v

# ============================================================================
# Code Quality
# ============================================================================

format: ## Format code with black
	black .

format-check: ## Check code formatting without making changes
	black --check .

lint: ## Lint code with ruff
	ruff check .

lint-fix: ## Lint and auto-fix issues with ruff
	ruff check --fix .

type-check: ## Run type checking with mypy (if installed)
	mypy pipeline/ --ignore-missing-imports || echo "mypy not installed"

quality: format lint ## Run all code quality checks

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Clean temporary files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build

clean-all: clean ## Clean everything including .env
	rm -f .env

# ============================================================================
# Docker
# ============================================================================

docker-build: ## Build Docker image
	docker build -t face-pipeline:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file ../.env face-pipeline:latest

docker-run-dev: ## Run Docker container in dev mode with volume mount
	docker run -p 8000:8000 --env-file ../.env -v $(PWD):/app face-pipeline:latest

# ============================================================================
# Development Helpers
# ============================================================================

shell: ## Start Python REPL with pipeline modules imported
	python -i -c "from pipeline import *; from config.settings import settings"

check-env: ## Verify environment configuration
	@echo "Checking environment configuration..."
	@python -c "from config.settings import settings; print('✓ Settings loaded successfully'); print(f'  MinIO: {settings.minio_endpoint}'); print(f'  Qdrant: {settings.qdrant_url}'); print(f'  Max Concurrent: {settings.max_concurrent}')"

deps-check: ## Check for outdated dependencies
	pip list --outdated

# ============================================================================
# CI/CD
# ============================================================================

ci: clean install test lint ## Run CI pipeline (clean, install, test, lint)

pre-commit: format lint test ## Run pre-commit checks (format, lint, test)

# ============================================================================
# Quick shortcuts
# ============================================================================

t: test ## Shortcut for 'make test'
f: format ## Shortcut for 'make format'
l: lint ## Shortcut for 'make lint'
r: run ## Shortcut for 'make run'
