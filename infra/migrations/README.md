# Database Migrations

This directory contains SQL migration files for the Mordeaux Face Protection System database schema.

## Migration Files

The migrations are numbered sequentially and should be run in order:

1. **001_create_tenants.sql** - Creates the tenants table for multi-tenant support
2. **002_create_sources.sql** - Creates the sources table for content sources
3. **003_create_content.sql** - Creates the content table for processed content
4. **004_create_faces.sql** - Creates the faces table for detected faces
5. **005_create_embeddings.sql** - Creates the embeddings table for face vectors
6. **006_create_clusters.sql** - Creates the clusters table for face clustering
7. **007_create_cluster_members.sql** - Creates the cluster_members table for cluster relationships
8. **008_create_policies.sql** - Creates the policies table for access control
9. **009_create_event_audit.sql** - Creates the event_audit table for system events
10. **010_create_alerts.sql** - Creates the alerts table for system alerts

## Running Migrations

### Using Make (Recommended)

```bash
# Run all migrations
make migrate

# Check migration status
make migrate-status

# Reset database (WARNING: drops all data)
make migrate-reset
```

### Using Migration Scripts Directly

#### Linux/macOS
```bash
# Run migrations
./infra/migrations/migrate.sh migrate

# Check status
./infra/migrations/migrate.sh status

# Reset database
./infra/migrations/migrate.sh reset
```

#### Windows PowerShell
```powershell
# Run migrations
.\infra\migrations\migrate.ps1 migrate

# Check status
.\infra\migrations\migrate.ps1 status

# Reset database
.\infra\migrations\migrate.ps1 reset
```

### Manual Migration

If the migration scripts are not available, you can run migrations manually:

```bash
# Start PostgreSQL container
docker-compose -f infra/docker-compose.yml up -d postgres

# Wait for PostgreSQL to be ready
docker-compose -f infra/docker-compose.yml exec postgres pg_isready -U postgres

# Run each migration file
docker-compose -f infra/docker-compose.yml exec -T postgres psql -U postgres -d mordeaux < infra/migrations/001_create_tenants.sql
docker-compose -f infra/docker-compose.yml exec -T postgres psql -U postgres -d mordeaux < infra/migrations/002_create_sources.sql
# ... continue for all migration files
```

## Database Schema Overview

### Core Tables

- **tenants** - Multi-tenant organization structure
- **sources** - Content sources (crawlers, uploads, APIs)
- **content** - Processed content items with metadata
- **faces** - Detected faces with bounding boxes and quality scores
- **embeddings** - Face embeddings (512-dimensional vectors)
- **clusters** - Face clusters for grouping similar faces
- **cluster_members** - Relationships between faces and clusters
- **policies** - Tenant-specific access control rules
- **event_audit** - System events and audit trail
- **alerts** - System alerts and notifications

### Key Features

- **UUID Primary Keys** - All tables use UUID primary keys for better distributed system support
- **Timestamps** - All tables include `created_at` and `updated_at` timestamps
- **JSONB Support** - Flexible metadata storage using PostgreSQL's JSONB type
- **Comprehensive Indexing** - Optimized indexes for common query patterns
- **Multi-tenant Ready** - Tenant isolation built into the schema
- **Audit Trail** - Complete event tracking and audit capabilities

### Indexes

Each table includes carefully designed indexes for:
- Primary key lookups
- Foreign key relationships
- Common query patterns
- Time-based queries
- Multi-tenant filtering
- Performance optimization

## Adding New Migrations

When adding new migrations:

1. **Number sequentially** - Use the next available number (e.g., 011_create_new_table.sql)
2. **Include rollback** - Consider adding a rollback script if needed
3. **Test thoroughly** - Test migrations on a copy of production data
4. **Document changes** - Update this README with new table descriptions
5. **Update indexes** - Add appropriate indexes for new tables

## Environment Variables

The migration scripts support the following environment variables:

- `DB_HOST` - PostgreSQL host (default: localhost)
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name (default: mordeaux)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password (default: postgres)

## Troubleshooting

### Common Issues

1. **PostgreSQL not ready** - Wait for the container to fully start
2. **Permission denied** - Ensure Docker containers have proper permissions
3. **Migration already exists** - Check if tables already exist before running
4. **Connection refused** - Verify PostgreSQL container is running and accessible

### Debug Mode

To debug migration issues:

```bash
# Check PostgreSQL logs
docker-compose -f infra/docker-compose.yml logs postgres

# Connect to database directly
docker-compose -f infra/docker-compose.yml exec postgres psql -U postgres -d mordeaux

# List all tables
\dt

# Describe a table
\d table_name
```

## Production Considerations

For production deployments:

1. **Backup first** - Always backup before running migrations
2. **Test migrations** - Test on staging environment first
3. **Monitor performance** - Watch for slow queries during migration
4. **Plan downtime** - Some migrations may require brief downtime
5. **Rollback plan** - Have a rollback strategy ready
