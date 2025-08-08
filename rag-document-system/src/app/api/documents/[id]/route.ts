import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse } from '@/lib/utils/response';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';
import { getDocumentById, deleteDocumentAndChunks } from '@/lib/database/documentQueries';

/**
 * GET /api/documents/:id
 * Get a single document by ID
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

    return successResponse({
      document: {
        id: document.id,
        title: document.title,
        file_name: document.file_name,
        file_type: document.file_type,
        file_size: document.file_size,
        content_preview: document.content_preview,
        processing_status: document.processing_status,
        chunk_count: document.chunk_count,
        processing_error: document.processing_error,
        file_metadata: document.file_metadata,
        created_at: document.created_at,
        updated_at: document.updated_at,
      },
    }, 'Document retrieved successfully');

  } catch (error) {
    console.error('Document get error:', error);
    
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

/**
 * DELETE /api/documents/:id
 * Delete a document and all its chunks
 */
export async function DELETE(
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

    // Delete document and all related data
    await deleteDocumentAndChunks(documentId, decoded.userId);

    return successResponse({
      message: 'Document deleted successfully',
      document_id: documentId,
    }, 'Document deleted successfully');

  } catch (error) {
    console.error('Document delete error:', error);
    
    if (error instanceof Error) {
      if (error.message.includes('expired')) {
        return unauthorizedResponse('Token has expired');
      }
      if (error.message.includes('invalid')) {
        return unauthorizedResponse('Invalid token');
      }
      if (error.message.includes('not found')) {
        return errorResponse('Document not found', 404);
      }
    }
    
    return errorResponse('Internal server error', 500);
  }
}
