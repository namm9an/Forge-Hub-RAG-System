# Backend

FastAPI backend managed by Poetry.

## Setup

- Install dependencies: `poetry install`
- Run dev server: `poetry run uvicorn app.api.main:app --reload`

## Environment

Configuration is loaded via Pydantic Settings from environment variables with prefix `APP_`.
Supported `.env` files: `.env`, `.env.local`.

Key variables:
- APP_ENV (development|staging|production|test)
- APP_DEBUG
- APP_HOST / APP_PORT
- APP_DATABASE_URL
- APP_SUPABASE_URL / APP_SUPABASE_ANON_KEY
