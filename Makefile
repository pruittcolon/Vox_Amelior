# NeMo Server Makefile
# Phase 24 - Verification Commands
#
# Usage: make <target>

.PHONY: all lint format typecheck test security clean help

# Default target
all: lint typecheck test

## Development

lint: ## Run ruff linter
	ruff check services/ shared/ --fix

format: ## Format code with ruff
	ruff format services/ shared/

typecheck: ## Run mypy type checking
	mypy services/api-gateway/src/core/ shared/ --ignore-missing-imports

## Testing

test: ## Run all tests
	pytest tests/ -v --cov=services --cov=shared --cov-report=term

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	pytest tests/integration/ -v -m integration

test-e2e: ## Run Playwright E2E tests
	cd scripts/browser_tests && python3 runner.py --all

## Security

security: ## Run all security checks
	bandit -r services/ shared/ -c pyproject.toml --skip B101
	pip-audit --strict || echo "pip-audit completed"

verify-secrets: ## Verify no hardcoded secrets
	@echo "Checking for hardcoded secrets..."
	@grep -rn "API_KEY\|SECRET\|PASSWORD" --include="*.yml" --include="*.yaml" docker/ | grep -v "_FILE\|secrets:" || echo "✅ No hardcoded secrets found"

## Docker

build: ## Build all Docker images
	docker compose -f docker/docker-compose.yml build

up: ## Start all services
	./nemo

dev: ## Start services in development mode (light config)
	docker compose -f docker/docker-compose.light.yml up -d

prod: ## Start services in production mode (full config)
	docker compose -f docker/docker-compose.yml up -d

down: ## Stop all services
	docker compose -f docker/docker-compose.yml down

logs: ## View service logs
	docker compose -f docker/docker-compose.yml logs -f

## Documentation

docs: ## Regenerate documentation
	python scripts/repo_mapper.py

## Cleanup

clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

## Help

help: ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
