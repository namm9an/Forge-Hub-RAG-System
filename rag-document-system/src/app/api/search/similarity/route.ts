import { NextRequest, NextResponse } from 'next/server';
import { similaritySearchService } from '@/lib/vectors/similaritySearch';
import { geminiEmbeddingService } from '@/lib/embeddings/geminiEmbeddings';
import { createApiResponse } from '@/lib/utils/response';
import { getAuthenticatedUser } from '@/lib/auth/utils';
import { SimilaritySearchRequest } from '@/lib/types';

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
      similarity_threshold = 0.7,
      limit = 20,
      offset = 0,
      distance_metric = 'cosine',
      include_metadata = true,
      filters = {}
    }: SimilaritySearchRequest = body;

    // Validate input
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return createApiResponse(null, 'Query is required', 400);
    }

    if (limit > 100) {
      return createApiResponse(null, 'Maximum limit is 100', 400);
    }

    // Generate query embedding
    const queryEmbedding = await geminiEmbeddingService.generateEmbedding({
      text: query.trim(),
      cache_result: true
    });

    // Build similarity search query
    const { createSupabaseClient } = await import('@/lib/supabase/client');
    const supabase = createSupabaseClient();

    let searchQuery = supabase
      .from('embeddings')
      .select(`
        id,
        embedding,
        model_version,
        processing_time_ms,
        created_at,
        document_chunks!inner (
          id,
          content,
          chunk_index,
          metadata,
          document_id,
          documents!inner (
            id,
            title,
            file_name,
            file_type,
            created_at,
            user_id
          )
        )
      `)
      .eq('document_chunks.documents.user_id', user.id)
      .eq('status', 'completed')
      .order('created_at', { ascending: false });

    // Apply filters
    if (filters.document_ids && filters.document_ids.length > 0) {
      searchQuery = searchQuery.in('document_chunks.document_id', filters.document_ids);
    }

    if (filters.file_types && filters.file_types.length > 0) {
      searchQuery = searchQuery.in('document_chunks.documents.file_type', filters.file_types);
    }

    if (filters.date_range) {
      if (filters.date_range.start) {
        searchQuery = searchQuery.gte('document_chunks.documents.created_at', filters.date_range.start);
      }
      if (filters.date_range.end) {
        searchQuery = searchQuery.lte('document_chunks.documents.created_at', filters.date_range.end);
      }
    }

    const { data: embeddings, error } = await searchQuery;

    if (error) {
      console.error('Error searching embeddings:', error);
      return createApiResponse(null, 'Error performing similarity search', 500);
    }

    if (!embeddings || embeddings.length === 0) {
      return createApiResponse({
        results: [],
        total: 0,
        query: query,
        similarity_threshold,
        execution_time_ms: 0
      });
    }

    // Calculate similarities and rank results
    const startTime = Date.now();
    const results = embeddings
      .map((embedding: any) => {
        const similarity = similaritySearchService.calculateCosineSimilarity(
          queryEmbedding.embedding,
          embedding.embedding
        );

        return {
          chunk: {
            id: embedding.document_chunks.id,
            document_id: embedding.document_chunks.document_id,
            chunk_index: embedding.document_chunks.chunk_index,
            content: embedding.document_chunks.content,
            metadata: embedding.document_chunks.metadata,
            created_at: embedding.document_chunks.created_at || ''
          },
          document: {
            id: embedding.document_chunks.documents.id,
            user_id: embedding.document_chunks.documents.user_id,
            title: embedding.document_chunks.documents.title,
            file_name: embedding.document_chunks.documents.file_name,
            file_type: embedding.document_chunks.documents.file_type,
            content_preview: '',
            processing_status: 'completed' as const,
            upload_path: '',
            created_at: embedding.document_chunks.documents.created_at,
            updated_at: embedding.document_chunks.documents.created_at
          },
          similarity_score: similarity,
          distance: 1 - similarity,
          metadata: include_metadata ? {
            chunk_index: embedding.document_chunks.chunk_index,
            model_version: embedding.model_version,
            processing_time_ms: embedding.processing_time_ms,
            embedding_created_at: embedding.created_at
          } : {},
          highlights: []
        };
      })
      .filter(result => result.similarity_score >= similarity_threshold)
      .sort((a, b) => b.similarity_score - a.similarity_score)
      .slice(offset, offset + limit);

    const executionTime = Date.now() - startTime;

    // Cache search results
    await cacheSearchResults(user.id, query, results, similarity_threshold);

    // Log search for analytics
    await logSearchQuery(user.id, query, results.length);

    return createApiResponse({
      results,
      total: results.length,
      query: query,
      similarity_threshold,
      execution_time_ms: executionTime,
      filters_applied: filters,
      model_version: queryEmbedding.model_version
    });

  } catch (error) {
    console.error('Error in similarity search:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
}

async function cacheSearchResults(
  userId: string,
  query: string,
  results: any[],
  similarityThreshold: number
) {
  try {
    const { createSupabaseClient } = await import('@/lib/supabase/client');
    const supabase = createSupabaseClient();

    const queryHash = await import('crypto').then(crypto => 
      crypto.createHash('sha256').update(query).digest('hex')
    );

    const expiresAt = new Date();
    expiresAt.setHours(expiresAt.getHours() + 1); // Cache for 1 hour

    await supabase
      .from('vector_search_cache')
      .upsert({
        user_id: userId,
        query_hash: queryHash,
        query_text: query,
        search_results: results,
        similarity_threshold: similarityThreshold,
        result_count: results.length,
        expires_at: expiresAt.toISOString()
      }, {
        onConflict: 'user_id,query_hash'
      });

  } catch (error) {
    console.error('Error caching search results:', error);
    // Don't throw - caching failure shouldn't break search
  }
}

async function logSearchQuery(userId: string, query: string, resultCount: number) {
  try {
    const { createSupabaseClient } = await import('@/lib/supabase/client');
    const supabase = createSupabaseClient();

    await supabase
      .from('search_history')
      .insert({
        user_id: userId,
        query: query,
        results_count: resultCount
      });

  } catch (error) {
    console.error('Error logging search query:', error);
    // Don't throw - logging failure shouldn't break search
  }
}
