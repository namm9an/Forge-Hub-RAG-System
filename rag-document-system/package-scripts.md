# Forge-Hub RAG System - Development Scripts

Add these scripts to your `package.json` file for easier development:

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "setup-db": "node setup-database.js",
    "test-api": "node test-api.js",
    "dev-full": "npm run setup-db && npm run dev",
    "check-health": "curl http://localhost:3000/api/health",
    "check-system": "curl http://localhost:3000/api/system/info"
  }
}
```

## Usage

1. **Setup Database**: `npm run setup-db`
2. **Start Development**: `npm run dev`
3. **Full Setup**: `npm run dev-full` (sets up DB and starts dev server)
4. **Test APIs**: `npm run test-api` (run this after starting dev server)
5. **Quick Health Check**: `npm run check-health`

## Development Workflow

1. First time setup:
   ```bash
   npm install
   npm run setup-db
   npm run dev
   ```

2. Daily development:
   ```bash
   npm run dev
   ```

3. Test your changes:
   ```bash
   npm run test-api
   ```

## Database Management

The database setup script will:
- Create all required tables
- Set up Row Level Security (RLS) policies
- Create indexes for better performance
- Add triggers for automatic timestamps

## API Testing

The API test script will verify:
- ✅ Health check endpoint
- ✅ System info endpoint  
- ✅ User registration
- ✅ User authentication
- ✅ Protected routes (documents, search)
- ✅ JWT token handling
