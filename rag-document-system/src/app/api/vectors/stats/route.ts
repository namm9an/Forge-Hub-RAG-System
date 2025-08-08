import { NextRequest, NextResponse } from 'next/server';
import { vectorStorageService } from '@/lib/vectors/vectorStorage';
import { indexManager } from '@/lib/vectors/indexManager';
import { createApiResponse } from '@/lib/utils/response';
import { getAuthenticatedUser } from '@/lib/auth/utils';

export async function GET(request: NextRequest) {
  try {
    // Authenticate user
    const user = await getAuthenticatedUser(request);
    if (!user) {
      return createApiResponse(null, 'Unauthorized', 401);
    }

    // Get vector statistics
    const stats = await vectorStorageService.getEmbeddingStats();
    
    // Get index health
    const indexHealth = await indexManager.getIndexHealth();
    
    // Get index usage monitoring
    const indexUsage = await indexManager.monitorIndexUsage();
    
    // Get index recommendations
    const recommendations = await indexManager.getIndexRecommendations();

    return createApiResponse({
      vector_stats: stats,
      index_health: indexHealth,
      index_usage: indexUsage,
      recommendations: recommendations,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error getting vector stats:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
}
