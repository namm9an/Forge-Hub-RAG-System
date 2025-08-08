# Forge-Hub RAG System - Phase 1 Foundation

A modern RAG (Retrieval-Augmented Generation) document system built with Next.js, Supabase, and Gemini 2.5.

## Phase 1 Features

✅ **Completed:**
- Next.js 14 with TypeScript and Tailwind CSS
- Supabase database integration with RLS policies
- JWT-based authentication system
- Rate limiting middleware
- Comprehensive API structure
- Gemini 2.5 API client setup
- Health check and system monitoring

## Tech Stack

- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **Backend:** Next.js API Routes
- **Database:** Supabase (PostgreSQL)
- **Authentication:** Supabase Auth + Custom JWT
- **AI:** Google Gemini 2.5 Flash
- **Deployment:** Vercel (frontend) + Render (backend)

## Setup Instructions

### 1. Database Setup

1. Go to your Supabase dashboard
2. Navigate to SQL Editor
3. Copy and paste the content from `src/lib/supabase/schema.sql`
4. Run the SQL script to create all tables and policies

### 2. Environment Variables

Update your `.env.local` file with your Gemini API key:

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

### 3. Install Dependencies

```bash
npm install
```

### 4. Run Development Server

```bash
npm run dev
```

Visit `http://localhost:3000` to see the application.

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Register new user
- `POST /api/auth/signin` - User login
- `GET /api/auth/me` - Get current user profile

### Documents
- `GET /api/documents` - List user documents
- `POST /api/documents/upload` - Upload document (stub)

### System
- `GET /api/health` - Health check
- `GET /api/system/info` - System information

## Database Schema

### Tables
- `profiles` - User profiles
- `documents` - Document metadata
- `document_chunks` - Document content chunks
- `embeddings` - Vector embeddings
- `search_history` - User search history

### Security
- Row Level Security (RLS) enabled on all tables
- Users can only access their own data
- Automatic profile creation on user signup

## Development

### Testing API Endpoints

You can test the API endpoints using tools like Postman or curl:

```bash
# Health check
curl http://localhost:3000/api/health

# System info
curl http://localhost:3000/api/system/info

# User signup
curl -X POST http://localhost:3000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "SecurePass123!", "full_name": "Test User"}'

# User signin
curl -X POST http://localhost:3000/api/auth/signin \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "SecurePass123!"}'
```

### Project Structure

```
src/
├── app/
│   ├── api/
│   │   ├── auth/          # Authentication endpoints
│   │   ├── documents/     # Document management
│   │   ├── health/        # Health check
│   │   └── system/        # System info
│   └── page.tsx           # Main page
├── lib/
│   ├── auth/              # Authentication utilities
│   ├── gemini/            # Gemini API client
│   ├── supabase/          # Database configuration
│   ├── types/             # TypeScript types
│   └── utils/             # Utility functions
└── middleware.ts          # Rate limiting & auth middleware
```

## Rate Limiting

- **Global:** 100 requests per 15 minutes
- **Auth:** 5 requests per minute
- **Upload:** 10 requests per hour

## Security Features

- JWT token authentication
- Password strength validation
- Rate limiting on all endpoints
- Row Level Security in database
- Input validation and sanitization
- CORS configuration

## Next Steps (Phase 2)

- [ ] Document processing (PDF, DOCX, TXT)
- [ ] File upload to Supabase Storage
- [ ] Text extraction and chunking
- [ ] Vector embedding generation
- [ ] Search functionality

## Cost Summary

**Total Cost: $0.00** (Free tier only)

- **Supabase:** Free tier (500MB database, 50MB storage)
- **Gemini 2.5:** Free tier (15 RPM, 1M tokens/day)
- **Vercel:** Free tier (100GB bandwidth)
- **Render:** Free tier (512MB RAM)

## License

This project is for educational purposes.
