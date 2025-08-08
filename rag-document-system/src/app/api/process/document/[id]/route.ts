import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse } from '@/lib/utils/response';
import { processNextJob } from '@/lib/processing/jobQueue';

/**
 * POST /api/process/document/:id
 * Internal endpoint to process a document
 * This would typically be called by a background worker
 */
export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    // In a production environment, you would verify service-to-service authentication
    // For now, we'll just process the next job from the queue
    
    await processNextJob();

    return successResponse({
      message: 'Processing job queued',
      document_id: params.id,
    }, 'Processing started successfully');

  } catch (error) {
    console.error('Document processing error:', error);
    return errorResponse('Internal server error', 500);
  }
}
