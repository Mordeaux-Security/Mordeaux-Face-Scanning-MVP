.PHONY: help up down logs ps migrate seed clean build test lint

# Default target
help: ## Show this help message
	@echo "Mordeaux Face Protection System"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

up: ## Start all services
	@echo "Starting Mordeaux services..."
	docker-compose -f infra/docker-compose.yml up -d
	@echo "Services started! Check status with 'make ps'"

down: ## Stop all services
	@echo "Stopping Mordeaux services..."
	docker-compose -f infra/docker-compose.yml down

logs: ## Show logs for all services
	docker-compose -f infra/docker-compose.yml logs -f

logs-api: ## Show logs for API Gateway
	docker-compose -f infra/docker-compose.yml logs -f api-gateway

logs-auth: ## Show logs for Auth service
	docker-compose -f infra/docker-compose.yml logs -f auth

logs-orchestrator: ## Show logs for Orchestrator
	docker-compose -f infra/docker-compose.yml logs -f orchestrator

logs-worker: ## Show logs for Face Worker
	docker-compose -f infra/docker-compose.yml logs -f face-worker

logs-vector-index: ## Show logs for Vector Index
	docker-compose -f infra/docker-compose.yml logs -f vector-index

logs-search-api: ## Show logs for Search API
	docker-compose -f infra/docker-compose.yml logs -f search-api

logs-policy-engine: ## Show logs for Policy Engine
	docker-compose -f infra/docker-compose.yml logs -f policy-engine

ps: ## Show running services
	docker-compose -f infra/docker-compose.yml ps

migrate: ## Run database migrations
	@echo "Running database migrations..."
	@if [ -f "infra/migrations/migrate.sh" ]; then \
		./infra/migrations/migrate.sh migrate; \
	elif [ -f "infra/migrations/migrate.ps1" ]; then \
		powershell -ExecutionPolicy Bypass -File infra/migrations/migrate.ps1 migrate; \
	else \
		echo "Migration script not found. Using fallback method..."; \
		docker-compose -f infra/docker-compose.yml exec postgres psql -U postgres -d mordeaux -c "SELECT 'No migrations found' as status;"; \
	fi

migrate-status: ## Show migration status
	@echo "Checking migration status..."
	@if [ -f "infra/migrations/migrate.sh" ]; then \
		./infra/migrations/migrate.sh status; \
	elif [ -f "infra/migrations/migrate.ps1" ]; then \
		powershell -ExecutionPolicy Bypass -File infra/migrations/migrate.ps1 status; \
	else \
		echo "Migration script not found. Using fallback method..."; \
		docker-compose -f infra/docker-compose.yml exec postgres psql -U postgres -d mordeaux -c "\dt"; \
	fi

migrate-reset: ## Reset database (WARNING: drops all data)
	@echo "WARNING: This will drop all data in the database!"
	@if [ -f "infra/migrations/migrate.sh" ]; then \
		./infra/migrations/migrate.sh reset; \
	elif [ -f "infra/migrations/migrate.ps1" ]; then \
		powershell -ExecutionPolicy Bypass -File infra/migrations/migrate.ps1 reset; \
	else \
		echo "Migration script not found. Using fallback method..."; \
		docker-compose -f infra/docker-compose.yml exec postgres psql -U postgres -d mordeaux -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; \
	fi

seed: ## Seed database with initial data
	@echo "Seeding database..."
	# TODO: Add seed data

clean: ## Clean up containers and volumes
	@echo "Cleaning up..."
	docker-compose -f infra/docker-compose.yml down -v
	docker system prune -f

build: ## Build all services
	@echo "Building services..."
	docker-compose -f infra/docker-compose.yml build

test: ## Run tests
	@echo "Running tests..."
	# TODO: Add test commands

lint: ## Run linting
	@echo "Running linting..."
	# TODO: Add lint commands

# Development helpers
dev-up: ## Start services in development mode
	@echo "Starting development environment..."
	docker-compose -f infra/docker-compose.yml up -d postgres redis rabbitmq minio vector-index
	@echo "Infrastructure services started. Start individual services manually for development."

dev-down: ## Stop development services
	docker-compose -f infra/docker-compose.yml stop postgres redis rabbitmq minio vector-index

# Service URLs
urls: ## Show service URLs
	@echo "Service URLs:"
	@echo "============="
	@echo "API Gateway:     http://localhost:3000"
	@echo "API Docs:        http://localhost:3000/docs"
	@echo "Auth Service:    http://localhost:3001"
	@echo "Upload Service:  http://localhost:3002"
	@echo "Orchestrator:    http://localhost:3003"
	@echo "Policy Engine:   http://localhost:3004"
	@echo "Search API:      http://localhost:3005"
	@echo "Search Docs:     http://localhost:3005/docs"
	@echo "Vector Index:    http://localhost:3006"
	@echo "Admin Console:   http://localhost:3007"
	@echo "MinIO Console:   http://localhost:9001"
	@echo "RabbitMQ Mgmt:   http://localhost:15672"
