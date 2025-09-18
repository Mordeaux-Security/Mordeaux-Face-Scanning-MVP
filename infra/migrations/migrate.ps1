# Migration script for Mordeaux Face Protection System (Windows PowerShell)
# This script applies all migration files in order using psql from Docker

param(
    [string]$Command = "migrate"
)

# Configuration
$DB_HOST = if ($env:DB_HOST) { $env:DB_HOST } else { "localhost" }
$DB_PORT = if ($env:DB_PORT) { $env:DB_PORT } else { "5432" }
$DB_NAME = if ($env:DB_NAME) { $env:DB_NAME } else { "mordeaux" }
$DB_USER = if ($env:DB_USER) { $env:DB_USER } else { "postgres" }
$DB_PASSWORD = if ($env:DB_PASSWORD) { $env:DB_PASSWORD } else { "postgres" }

# Colors for output
$RED = "Red"
$GREEN = "Green"
$YELLOW = "Yellow"
$BLUE = "Blue"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $BLUE
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $GREEN
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $YELLOW
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $RED
}

# Function to check if PostgreSQL is ready
function Wait-ForPostgres {
    Write-Status "Waiting for PostgreSQL to be ready..."
    
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $result = docker exec mordeaux-postgres-1 pg_isready -h localhost -p 5432 -U postgres 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "PostgreSQL is ready!"
                return $true
            }
        }
        catch {
            # Ignore errors and continue
        }
        
        Write-Status "Attempt $attempt/$maxAttempts - PostgreSQL not ready yet, waiting 2 seconds..."
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    Write-Error "PostgreSQL failed to become ready after $maxAttempts attempts"
    return $false
}

# Function to run a migration file
function Invoke-Migration {
    param([string]$MigrationFile)
    
    $migrationName = [System.IO.Path]::GetFileNameWithoutExtension($MigrationFile)
    Write-Status "Running migration: $migrationName"
    
    try {
        Get-Content $MigrationFile | docker exec -i mordeaux-postgres-1 psql -h localhost -p 5432 -U postgres -d mordeaux
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Migration $migrationName completed successfully"
            return $true
        } else {
            Write-Error "Migration $migrationName failed"
            return $false
        }
    }
    catch {
        Write-Error "Migration $migrationName failed with error: $_"
        return $false
    }
}

# Function to run all migrations
function Start-Migrations {
    Write-Status "Starting database migrations..."
    
    # Check if we're in the right directory
    if (-not (Test-Path "infra/migrations")) {
        Write-Error "Migration directory not found. Please run this script from the project root."
        exit 1
    }
    
    # Wait for PostgreSQL to be ready
    if (-not (Wait-ForPostgres)) {
        exit 1
    }
    
    # Get list of migration files in order
    $migrationFiles = Get-ChildItem "infra/migrations/*.sql" | Sort-Object Name
    
    if ($migrationFiles.Count -eq 0) {
        Write-Warning "No migration files found"
        return
    }
    
    Write-Status "Found $($migrationFiles.Count) migration files"
    
    # Run each migration
    $successCount = 0
    $totalCount = $migrationFiles.Count
    
    foreach ($migrationFile in $migrationFiles) {
        if (Invoke-Migration $migrationFile.FullName) {
            $successCount++
        } else {
            Write-Error "Migration failed. Stopping migration process."
            exit 1
        }
    }
    
    Write-Success "All migrations completed successfully! ($successCount/$totalCount)"
}

# Function to show migration status
function Show-MigrationStatus {
    Write-Status "Checking migration status..."
    
    try {
        $result = docker exec mordeaux-postgres-1 psql -h localhost -p 5432 -U postgres -d mordeaux -c "\dt" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Current database tables:"
            Write-Host $result
        } else {
            Write-Warning "Database connection failed or no tables found"
        }
    }
    catch {
        Write-Warning "Database connection failed or no tables found"
    }
}

# Function to reset database
function Reset-Database {
    Write-Warning "This will DROP ALL TABLES and data in the database!"
    $confirm = Read-Host "Are you sure you want to continue? (yes/no)"
    
    if ($confirm -ne "yes") {
        Write-Status "Database reset cancelled"
        return
    }
    
    Write-Status "Resetting database..."
    
    try {
        $resetScript = @"
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO public;
"@
        
        $resetScript | docker exec -i mordeaux-postgres-1 psql -h localhost -p 5432 -U postgres -d mordeaux
        Write-Success "Database reset completed"
    }
    catch {
        Write-Error "Database reset failed: $_"
    }
}

# Function to show help
function Show-Help {
    Write-Host "Usage: .\migrate.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  migrate  Run all pending migrations (default)"
    Write-Host "  status   Show current database status"
    Write-Host "  reset    Reset database (WARNING: drops all data)"
    Write-Host "  help     Show this help message"
    Write-Host ""
    Write-Host "Environment variables:"
    Write-Host "  DB_HOST     PostgreSQL host (default: localhost)"
    Write-Host "  DB_PORT     PostgreSQL port (default: 5432)"
    Write-Host "  DB_NAME     Database name (default: mordeaux)"
    Write-Host "  DB_USER     Database user (default: postgres)"
    Write-Host "  DB_PASSWORD Database password (default: postgres)"
}

# Main script logic
switch ($Command.ToLower()) {
    "migrate" {
        Start-Migrations
    }
    "status" {
        Show-MigrationStatus
    }
    "reset" {
        Reset-Database
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host "Use '.\migrate.ps1 help' for usage information"
        exit 1
    }
}
