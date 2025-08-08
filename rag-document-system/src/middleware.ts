import { NextRequest, NextResponse } from 'next/server';

// Simple in-memory rate limiter for development
const rateLimitStore = new Map<string, { count: number; resetTime: number }>();

function checkRateLimit(key: string, limit: number, windowMs: number): boolean {
  const now = Date.now();
  const record = rateLimitStore.get(key);
  
  if (!record || now > record.resetTime) {
    rateLimitStore.set(key, { count: 1, resetTime: now + windowMs });
    return true;
  }
  
  if (record.count >= limit) {
    return false;
  }
  
  record.count++;
  return true;
}

// Rate limiting middleware
export async function rateLimitMiddleware(req: NextRequest) {
  const url = req.nextUrl.pathname;
  const clientIP = req.ip || 'unknown';

  let limit = 100;
  let windowMs = 15 * 60 * 1000; // 15 minutes

  if (url.startsWith('/api/auth')) {
    limit = 5;
    windowMs = 60 * 1000; // 1 minute
  } else if (url.startsWith('/api/documents/upload')) {
    limit = 10;
    windowMs = 60 * 60 * 1000; // 1 hour
  }

  const key = `${clientIP}:${url.split('/').slice(0, 3).join('/')}`;
  
  if (!checkRateLimit(key, limit, windowMs)) {
    return NextResponse.json(
      { success: false, error: 'Rate limit exceeded' },
      { status: 429 }
    );
  }

  return NextResponse.next();
}

// Main middleware function for Next.js
export function middleware(req: NextRequest) {
  // Skip middleware for static files and internal Next.js routes
  if (
    req.nextUrl.pathname.startsWith('/_next') ||
    req.nextUrl.pathname.startsWith('/api/health') ||
    req.nextUrl.pathname.startsWith('/api/system')
  ) {
    return NextResponse.next();
  }

  // Apply rate limiting to API routes
  if (req.nextUrl.pathname.startsWith('/api/')) {
    return rateLimitMiddleware(req);
  }

  return NextResponse.next();
}

// Configure which routes to run middleware on
export const config = {
  matcher: [
    '/(api/.*)'
  ]
};
