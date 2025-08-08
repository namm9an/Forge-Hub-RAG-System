import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse } from '@/lib/utils/response';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';
import { getDocumentById, updateDocumentStatus } from '@/lib/database/documentQueries';
import { addProcessingJob } from '@/lib/processing/jobQueue';
import { deleteDocumentChunks } from '@/lib/database/chunkQueries';

/**
 * POST /api/documents/:id/reprocess
 * Retry processing a failed document
 */
export async function POST(
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

    // Check if document can be reprocessed
    if (document.processing_status !== 'failed') {
      return errorResponse('Document is not in failed state', 400);
    }

    // Clean up existing chunks if any
    await deleteDocumentChunks(documentId);

    // Reset document status
    await updateDocumentStatus(documentId, 'pending', {
      processing_error: null,
      chunk_count: 0,
      content_preview: null,
    });

    // Add new processing job
    await addProcessingJob(documentId);

    return successResponse({
      message: 'Document reprocessing started',
      document: {
        id: document.id,
        title: document.title,
        processing_status: 'pending',
      },
    }, 'Document reprocessing started successfully');

  } catch (error) {
    console.error('Document reprocess error:', error);
    
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
