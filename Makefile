.PHONY: help up down logs ps migrate seed clean build test lint

# Default target
help: ## Show this help message
	@echo "Mordeaux Face Protection System"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

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

logs-worker: ## Show logs for Face Worker
	docker-compose -f infra/docker-compose.yml logs -f face-worker

ps: ## Show running services
	docker-compose -f infra/docker-compose.yml ps

migrate: ## Run database migrations
	@echo "Running database migrations..."
	docker-compose -f infra/docker-compose.yml exec postgres psql -U postgres -d mordeaux -f /docker-entrypoint-initdb.d/001_initial_schema.sql

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
	@echo "Admin Console:   http://localhost:3005"
	@echo "MinIO Console:   http://localhost:9001"
	@echo "RabbitMQ Mgmt:   http://localhost:15672"
	@echo "Vector Index:    http://localhost:8080"
