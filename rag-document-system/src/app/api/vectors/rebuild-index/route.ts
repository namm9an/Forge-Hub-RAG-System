import { NextRequest, NextResponse } from 'next/server';
import { indexManager } from '@/lib/vectors/indexManager';
import { createApiResponse } from '@/lib/utils/response';
import { getAuthenticatedUser } from '@/lib/auth/utils';

export async function POST(request: NextRequest) {
  try {
    // Authenticate user
    const user = await getAuthenticatedUser(request);
    if (!user) {
      return createApiResponse(null, 'Unauthorized', 401);
    }

    // Parse request body for options
    const body = await request.json().catch(() => ({}));
    const { force_rebuild = false, auto_optimize = true } = body;

    const startTime = Date.now();

    if (force_rebuild) {
      // Drop and recreate indexes
      await indexManager.dropAndRecreateIndexes();
    } else if (auto_optimize) {
      // Auto-optimize based on usage
      await indexManager.autoOptimizeIndexes();
    } else {
      // Standard rebuild
      await indexManager.rebuildVectorIndex();
    }

    const executionTime = Date.now() - startTime;

    // Get updated stats
    const stats = await indexManager.monitorIndexUsage();
    const health = await indexManager.getIndexHealth();

    return createApiResponse({
      message: 'Vector indexes rebuilt successfully',
      execution_time_ms: executionTime,
      operation: force_rebuild ? 'force_rebuild' : (auto_optimize ? 'auto_optimize' : 'standard_rebuild'),
      index_stats: stats,
      index_health: health,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error rebuilding vector indexes:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
}
