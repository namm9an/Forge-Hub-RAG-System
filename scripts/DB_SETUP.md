# Database Setup Instructions

## Prerequisites
- PostgreSQL 14+ with pgvector extension support
- psql command-line tool
- Database connection URL

## Setup Steps

### 1. Set Database URL
```bash
# Linux/Mac
export VECTOR_DB_URL="postgresql://username:password@localhost:5432/ragdb"

# Windows PowerShell
$env:VECTOR_DB_URL = "postgresql://username:password@localhost:5432/ragdb"
```

### 2. Run Bootstrap Script
```bash
psql $VECTOR_DB_URL -f scripts/bootstrap.sql
```

### 3. Validate Setup
```bash
psql $VECTOR_DB_URL -f scripts/validate_db.sql
```

Expected output:
- UUID generation test should return a valid UUID
- Vector test should return '[(0)]'
- Both extensions should be listed
- All 5 tables should be created
- Test session should be created and deleted
- Final status should show "Database validation successful!"

### 4. Alternative: Use Python Script
If psql is not available, the database is also bootstrapped automatically when the backend starts:

```bash
cd advanced-rag-chatbot/apps/backend
python -m uvicorn app.main:app --reload
```

The `ensure_schema()` function in `app/db.py` will run on startup.

## Troubleshooting

### pgvector not found
If you get an error about pgvector extension not found:
1. Install pgvector: https://github.com/pgvector/pgvector#installation
2. Or use Supabase/Neon which have pgvector pre-installed

### pgcrypto not found
This is a standard PostgreSQL extension and should be available. If not:
```sql
-- Check available extensions
SELECT * FROM pg_available_extensions WHERE name = 'pgcrypto';
```

### Connection refused
- Check PostgreSQL is running
- Verify connection string format
- Check firewall/network settings

## Testing the Extensions

### Test pgcrypto:
```sql
SELECT gen_random_uuid();
```

### Test vector:
```sql
SELECT '[(0,1,2)]'::vector;
SELECT '[(0.1, 0.2, 0.3)]'::vector(3);
```

### Test vector similarity:
```sql
SELECT 1 - (
  '[(1,0,0)]'::vector <=> '[(0,1,0)]'::vector
) AS cosine_similarity;
```
