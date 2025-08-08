import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse } from '@/lib/utils/response';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';
import { getQueueStatus } from '@/lib/processing/jobQueue';

/**
 * GET /api/process/queue
 * Get processing queue status
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

    // Get queue status
    const queueStatus = await getQueueStatus();

    return successResponse({
      queue: queueStatus,
      timestamp: new Date().toISOString(),
    }, 'Queue status retrieved successfully');

  } catch (error) {
    console.error('Queue status error:', error);
    
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
