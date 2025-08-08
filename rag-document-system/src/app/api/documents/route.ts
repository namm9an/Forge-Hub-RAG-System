import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse } from '@/lib/utils/response';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';
import { getDocumentsByUser } from '@/lib/database/documentQueries';

/**
 * GET /api/documents
 * List user documents with filtering and pagination
 */
export async function GET(request: NextRequest) {
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

    // Parse query parameters
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get('page') || '1', 10);
    const limit = parseInt(searchParams.get('limit') || '20', 10);
    const status = searchParams.get('status') as 'pending' | 'processing' | 'completed' | 'failed' | null;
    const fileType = searchParams.get('fileType');
    const searchQuery = searchParams.get('search');
    const orderBy = searchParams.get('orderBy') as 'created_at' | 'updated_at' | 'title' | 'file_size' || 'created_at';
    const orderDirection = searchParams.get('orderDirection') as 'asc' | 'desc' || 'desc';

    // Build filter
    const filter = {
      ...(status && { status }),
      ...(fileType && { fileType }),
      ...(searchQuery && { searchQuery }),
    };

    // Get documents
    const result = await getDocumentsByUser(
      decoded.userId,
      filter,
      {
        page,
        limit: Math.min(limit, 100), // Max 100 per page
        orderBy,
        orderDirection,
      }
    );

    return successResponse({
      documents: result.documents.map(doc => ({
        id: doc.id,
        title: doc.title,
        file_name: doc.file_name,
        file_type: doc.file_type,
        file_size: doc.file_size,
        content_preview: doc.content_preview,
        processing_status: doc.processing_status,
        chunk_count: doc.chunk_count,
        processing_error: doc.processing_error,
        created_at: doc.created_at,
        updated_at: doc.updated_at,
      })),
      pagination: {
        page: result.page,
        pageSize: result.pageSize,
        total: result.total,
        totalPages: result.totalPages,
        hasNext: result.page < result.totalPages,
        hasPrev: result.page > 1,
      },
    }, 'Documents retrieved successfully');

  } catch (error) {
    console.error('Documents list error:', error);
    
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
