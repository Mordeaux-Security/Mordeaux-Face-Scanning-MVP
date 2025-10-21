# Mordeaux Face Scanning MVP - Makefile
# Provides convenient commands for Docker operations

.PHONY: help build start stop restart status logs cleanup test

# Default target
help:
	@echo "🐳 Mordeaux Face Scanning MVP - Docker Commands"
	@echo "=============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make build     - Build and start all services"
	@echo "  make start     - Start all services"
	@echo "  make stop      - Stop all services"
	@echo "  make restart   - Restart all services"
	@echo "  make status    - Show service status"
	@echo "  make logs      - Show recent logs"
	@echo "  make cleanup   - Stop services and clean up resources"
	@echo "  make test      - Test Docker configuration"
	@echo "  make health    - Quick health check"
	@echo "  make smoketest - Comprehensive proxy smoke tests"
	@echo "  make smoketest-quick - Quick smoke tests"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Service URLs:"
	@echo "  Frontend:      http://localhost:3000"
	@echo "  Backend API:   http://localhost:8000"
	@echo "  Face Pipeline: http://localhost:8001"
	@echo "  MinIO Console: http://localhost:9001"
	@echo "  pgAdmin:       http://localhost:5050"
	@echo "  Qdrant:        http://localhost:6333"
	@echo "  Nginx:         http://localhost:80"

# Build and start all services
build:
	@echo "🚀 Building and starting all services..."
	docker-compose up --build -d
	@echo "✅ All services started successfully!"
	@echo ""
	@echo "Service URLs:"
	@echo "  🌐 Frontend:        http://localhost:3000"
	@echo "  🔧 Backend API:     http://localhost:8000"
	@echo "  🧠 Face Pipeline:   http://localhost:8001"
	@echo "  🗄️  MinIO Console:   http://localhost:9001"
	@echo "  📊 pgAdmin:         http://localhost:5050"
	@echo "  🔍 Qdrant:          http://localhost:6333"
	@echo "  🌍 Nginx (Main):    http://localhost:80"

# Start all services
start:
	@echo "▶️  Starting all services..."
	docker-compose up -d
	@echo "✅ All services started!"

# Stop all services
stop:
	@echo "⏹️  Stopping all services..."
	docker-compose down
	@echo "✅ All services stopped!"

# Restart all services
restart: stop start
	@echo "🔄 All services restarted!"

# Show service status
status:
	@echo "📊 Service Status:"
	docker-compose ps

# Show recent logs
logs:
	@echo "📋 Recent logs:"
	docker-compose logs --tail=50

# Clean up resources
cleanup:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup completed!"

# Test Docker configuration
test:
	@echo "🧪 Testing Docker configuration..."
	docker-compose config --quiet
	@echo "✅ Docker configuration is valid!"

# Development helpers
dev-logs:
	@echo "📋 Following logs (Ctrl+C to stop):"
	docker-compose logs -f

dev-shell-backend:
	@echo "🐚 Opening shell in backend container..."
	docker-compose exec backend-cpu bash

dev-shell-face-pipeline:
	@echo "🐚 Opening shell in face-pipeline container..."
	docker-compose exec face-pipeline bash

dev-shell-worker:
	@echo "🐚 Opening shell in worker container..."
	docker-compose exec worker-cpu bash

# Database helpers
db-backup:
	@echo "💾 Creating database backup..."
	docker-compose exec postgres pg_dump -U mordeaux mordeaux > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backup created!"

db-restore:
	@echo "📥 Restoring database from backup..."
	@read -p "Enter backup filename: " backup; \
	docker-compose exec -T postgres psql -U mordeaux mordeaux < $$backup
	@echo "✅ Database restored!"

# Monitoring
monitor:
	@echo "📊 Container resource usage:"
	docker stats --no-stream

# Quick health check
health:
	@echo "🏥 Health check:"
	@echo "Backend API:"
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "❌ Backend not responding"
	@echo ""
	@echo "Face Pipeline:"
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health || echo "❌ Face Pipeline not responding"
	@echo ""
	@echo "Frontend:"
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 || echo "❌ Frontend not responding"

# Comprehensive smoke tests
smoketest:
	@echo "🧪 Running comprehensive smoke tests..."
	@echo "This will test Nginx routing, CORS headers, port mapping, and API endpoints."
	@echo ""
	@if [ -f "scripts/smoke_test.sh" ]; then \
		bash scripts/smoke_test.sh; \
	elif [ -f "scripts/smoke_test.ps1" ]; then \
		powershell -ExecutionPolicy Bypass -File scripts/smoke_test.ps1; \
	else \
		echo "❌ No smoke test script found"; \
		exit 1; \
	fi

# Quick smoke test (Windows PowerShell)
smoketest-win:
	@echo "🧪 Running smoke tests (Windows)..."
	powershell -ExecutionPolicy Bypass -File scripts/smoke_test.ps1

# Quick smoke test (Linux/Mac)
smoketest-linux:
	@echo "🧪 Running smoke tests (Linux/Mac)..."
	bash scripts/smoke_test.sh

# Quick smoke test (simplified)
smoketest-quick:
	@echo "🧪 Running quick smoke tests..."
	@if [ -f "scripts/quick_smoke_test.ps1" ]; then \
		powershell -ExecutionPolicy Bypass -File scripts/quick_smoke_test.ps1; \
	else \
		echo "❌ Quick smoke test script not found"; \
		exit 1; \
	fi