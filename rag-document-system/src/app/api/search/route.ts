import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse, validationErrorResponse } from '@/lib/utils/response';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';
import { ValidationError } from '@/lib/types';

/**
 * POST /api/search
 * Search documents endpoint (stub for Phase 1)
 */
export async function POST(request: NextRequest) {
  try {
    const token = extractTokenFromHeader(request.headers.get('authorization'));
    if (!token) {
      return unauthorizedResponse('Authorization token is required');
    }

    // Verify JWT token
    const decoded = verifyJWT(token);
    if (!decoded.userId) {
      return unauthorizedResponse('Invalid token');
    }

    const { query, limit = 10, offset = 0 } = await request.json();

    // Validate input
    const errors: ValidationError[] = [];

    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      errors.push({ field: 'query', message: 'Search query is required' });
    }

    if (errors.length > 0) {
      return validationErrorResponse(errors);
    }

    // For Phase 1, return a stub response
    // In later phases, this will perform vector search
    return successResponse({
      query: query.trim(),
      results: [],
      total: 0,
      pagination: {
        limit,
        offset,
        hasMore: false,
      },
      message: 'Search functionality will be implemented in Phase 3',
      note: 'This is a Phase 1 stub. Vector search will be implemented in Phase 3.',
    });

  } catch (error) {
    console.error('Search error:', error);
    
    if (error instanceof Error) {
      if (error.message.includes('expired')) {
        return unauthorizedResponse('Token has expired');
      }
      if (error.message.includes('invalid')) {
        return unauthorizedResponse('Invalid token');
      }
    }
    
    return errorResponse('Internal server error', 500);
  }
}
