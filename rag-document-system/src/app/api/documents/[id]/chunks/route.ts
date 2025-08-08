import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse } from '@/lib/utils/response';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';
import { getDocumentById } from '@/lib/database/documentQueries';
import { getChunksByDocument } from '@/lib/database/chunkQueries';

/**
 * GET /api/documents/:id/chunks
 * Get document chunks with pagination
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

    // Verify document ownership
    const document = await getDocumentById(documentId, decoded.userId);
    if (!document) {
      return errorResponse('Document not found', 404);
    }

    // Parse query parameters
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get('page') || '1', 10);
    const limit = parseInt(searchParams.get('limit') || '50', 10);
    const offset = (page - 1) * limit;

    // Get chunks
    const chunks = await getChunksByDocument(documentId, limit, offset);

    // Calculate total pages (we need to get total count)
    const totalChunks = document.chunk_count || 0;
    const totalPages = Math.ceil(totalChunks / limit);

    return successResponse({
      document: {
        id: document.id,
        title: document.title,
        processing_status: document.processing_status,
        chunk_count: document.chunk_count,
      },
      chunks: chunks.map(chunk => ({
        id: chunk.id,
        chunk_index: chunk.chunk_index,
        content: chunk.content,
        metadata: chunk.metadata,
        created_at: chunk.created_at,
      })),
      pagination: {
        page,
        limit,
        total: totalChunks,
        totalPages,
        hasNext: page < totalPages,
        hasPrev: page > 1,
      },
    }, 'Document chunks retrieved successfully');

  } catch (error) {
    console.error('Document chunks error:', error);
    
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
