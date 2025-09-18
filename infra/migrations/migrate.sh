#!/bin/bash

# Migration script for Mordeaux Face Protection System
# This script applies all migration files in order using psql from Docker

set -e

# Configuration
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-mordeaux}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-postgres}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if PostgreSQL is ready
wait_for_postgres() {
    print_status "Waiting for PostgreSQL to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec mordeaux-postgres-1 pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; then
            print_success "PostgreSQL is ready!"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - PostgreSQL not ready yet, waiting 2 seconds..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "PostgreSQL failed to become ready after $max_attempts attempts"
    return 1
}

# Function to run a migration file
run_migration() {
    local migration_file=$1
    local migration_name=$(basename "$migration_file" .sql)
    
    print_status "Running migration: $migration_name"
    
    if docker exec -i mordeaux-postgres-1 psql -h localhost -p 5432 -U postgres -d mordeaux < "$migration_file"; then
        print_success "Migration $migration_name completed successfully"
        return 0
    else
        print_error "Migration $migration_name failed"
        return 1
    fi
}

# Main migration function
run_migrations() {
    print_status "Starting database migrations..."
    
    # Check if we're in the right directory
    if [ ! -d "infra/migrations" ]; then
        print_error "Migration directory not found. Please run this script from the project root."
        exit 1
    fi
    
    # Wait for PostgreSQL to be ready
    wait_for_postgres
    
    # Get list of migration files in order
    local migration_files=($(ls infra/migrations/*.sql | sort))
    
    if [ ${#migration_files[@]} -eq 0 ]; then
        print_warning "No migration files found"
        return 0
    fi
    
    print_status "Found ${#migration_files[@]} migration files"
    
    # Run each migration
    local success_count=0
    local total_count=${#migration_files[@]}
    
    for migration_file in "${migration_files[@]}"; do
        if run_migration "$migration_file"; then
            success_count=$((success_count + 1))
        else
            print_error "Migration failed. Stopping migration process."
            exit 1
        fi
    done
    
    print_success "All migrations completed successfully! ($success_count/$total_count)"
}

# Function to show migration status
show_status() {
    print_status "Checking migration status..."
    
    if ! docker exec mordeaux-postgres-1 psql -h localhost -p 5432 -U postgres -d mordeaux -c "\dt" >/dev/null 2>&1; then
        print_warning "Database connection failed or no tables found"
        return 1
    fi
    
    print_status "Current database tables:"
    docker exec mordeaux-postgres-1 psql -h localhost -p 5432 -U postgres -d mordeaux -c "\dt"
}

# Function to reset database (WARNING: This will drop all data!)
reset_database() {
    print_warning "This will DROP ALL TABLES and data in the database!"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        print_status "Database reset cancelled"
        return 0
    fi
    
    print_status "Resetting database..."
    
    # Drop all tables
    docker exec mordeaux-postgres-1 psql -h localhost -p 5432 -U postgres -d mordeaux -c "
        DROP SCHEMA public CASCADE;
        CREATE SCHEMA public;
        GRANT ALL ON SCHEMA public TO postgres;
        GRANT ALL ON SCHEMA public TO public;
    "
    
    print_success "Database reset completed"
}

# Main script logic
case "${1:-migrate}" in
    "migrate")
        run_migrations
        ;;
    "status")
        show_status
        ;;
    "reset")
        reset_database
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  migrate  Run all pending migrations (default)"
        echo "  status   Show current database status"
        echo "  reset    Reset database (WARNING: drops all data)"
        echo "  help     Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  DB_HOST     PostgreSQL host (default: localhost)"
        echo "  DB_PORT     PostgreSQL port (default: 5432)"
        echo "  DB_NAME     Database name (default: mordeaux)"
        echo "  DB_USER     Database user (default: postgres)"
        echo "  DB_PASSWORD Database password (default: postgres)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
