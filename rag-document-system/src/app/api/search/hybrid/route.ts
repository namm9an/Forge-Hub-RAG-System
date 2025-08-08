import { NextRequest, NextResponse } from 'next/server';
import { similaritySearchService } from '@/lib/vectors/similaritySearch';
import { createApiResponse } from '@/lib/utils/response';
import { getAuthenticatedUser } from '@/lib/auth/utils';
import { createSupabaseClient } from '@/lib/supabase/client';

export async function POST(request: NextRequest) {
  try {
    // Authenticate user
    const user = await getAuthenticatedUser(request);
    if (!user) {
      return createApiResponse(null, 'Unauthorized', 401);
    }

    // Parse request body
    const body = await request.json();
    const {
      query,
      semantic_weight = 0.7,
      keyword_weight = 0.3,
      similarity_threshold = 0.7,
      limit = 20,
      offset = 0,
      filters = {}
    } = body;

    // Validate input
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return createApiResponse(null, 'Query is required', 400);
    }

    if (limit > 100) {
      return createApiResponse(null, 'Maximum limit is 100', 400);
    }

    if (semantic_weight + keyword_weight !== 1) {
      return createApiResponse(null, 'Semantic and keyword weights must sum to 1', 400);
    }

    // Perform hybrid search
    const results = await similaritySearchService.performHybridSearch({
      query: query.trim(),
      semantic_weight,
      keyword_weight,
      similarity_threshold,
      limit,
      offset,
      filters
    });

    // Log search query
    await logSearchQuery(user.id, query, results.length);

    return createApiResponse({
      query,
      results,
      total_results: results.length,
      search_weights: {
        semantic: semantic_weight,
        keyword: keyword_weight
      },
      filters_applied: filters,
      search_metadata: {
        timestamp: new Date().toISOString(),
        user_id: user.id
      }
    });

  } catch (error) {
    console.error('Error performing hybrid search:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
}

async function logSearchQuery(userId: string, query: string, resultCount: number) {
  try {
    const supabase = createSupabaseClient();
    
    await supabase
      .from('search_history')
      .insert({
        user_id: userId,
        query: query.substring(0, 500), // Limit query length
        results_count: resultCount,
        search_type: 'hybrid'
      });
  } catch (error) {
    console.error('Error logging search query:', error);
  }
} 