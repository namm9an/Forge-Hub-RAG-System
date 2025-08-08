import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse } from '@/lib/utils/response';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';
import { getDocumentById } from '@/lib/database/documentQueries';

/**
 * GET /api/documents/:id/status
 * Get document processing status
 */
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
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

    const documentId = params.id;

    // Get document
    const document = await getDocumentById(documentId, decoded.userId);
    if (!document) {
      return errorResponse('Document not found', 404);
    }

    // Calculate processing progress
    let progress = 0;
    switch (document.processing_status) {
      case 'pending':
        progress = 0;
        break;
      case 'processing':
        progress = 50;
        break;
      case 'completed':
        progress = 100;
        break;
      case 'failed':
        progress = 0;
        break;
    }

    return successResponse({
      status: document.processing_status,
      progress,
      chunk_count: document.chunk_count || 0,
      processing_error: document.processing_error,
      content_preview: document.content_preview,
      file_metadata: document.file_metadata,
      updated_at: document.updated_at,
    }, 'Document status retrieved successfully');

  } catch (error) {
    console.error('Document status error:', error);
    
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
