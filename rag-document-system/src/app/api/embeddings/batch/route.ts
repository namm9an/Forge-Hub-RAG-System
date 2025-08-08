import { NextRequest, NextResponse } from 'next/server';
import { embeddingProcessor } from '@/lib/embeddings/embeddingProcessor';
import { createApiResponse } from '@/lib/utils/response';
import { getAuthenticatedUser } from '@/lib/auth/utils';
import { EmbeddingBatchRequest } from '@/lib/types';

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
      document_ids, 
      batch_size = 10, 
      priority = 'normal', 
      force_reprocess = false 
    }: EmbeddingBatchRequest & { force_reprocess?: boolean } = body;

    // Validate input
    if (!document_ids || !Array.isArray(document_ids) || document_ids.length === 0) {
      return createApiResponse(null, 'document_ids array is required', 400);
    }

    if (document_ids.length > 50) {
      return createApiResponse(null, 'Maximum 50 documents allowed per batch', 400);
    }

    // Validate documents exist and belong to user
    const { createSupabaseClient } = await import('@/lib/supabase/client');
    const supabase = createSupabaseClient();
    
    const { data: documents, error: docError } = await supabase
      .from('documents')
      .select('id, title, processing_status')
      .in('id', document_ids)
      .eq('user_id', user.id);

    if (docError) {
      console.error('Error validating documents:', docError);
      return createApiResponse(null, 'Error validating documents', 500);
    }

    if (!documents || documents.length === 0) {
      return createApiResponse(null, 'No valid documents found', 404);
    }

    // Check if all documents are processed
    const unprocessedDocs = documents.filter(doc => doc.processing_status !== 'completed');
    if (unprocessedDocs.length > 0) {
      return createApiResponse(null, 
        `Documents must be processed before generating embeddings: ${unprocessedDocs.map(d => d.title).join(', ')}`, 
        400
      );
    }

    // Process batch request
    const batchRequest: EmbeddingBatchRequest = {
      document_ids: documents.map(d => d.id),
      batch_size,
      priority,
      force_reprocess
    };

    const batchResponse = await embeddingProcessor.processBatchRequest(batchRequest);
    
    return createApiResponse({
      ...batchResponse,
      message: 'Batch embedding generation started',
      documents: documents.map(doc => ({
        id: doc.id,
        title: doc.title,
        status: 'queued'
      }))
    });

  } catch (error) {
    console.error('Error processing batch embeddings:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    // Authenticate user
    const user = await getAuthenticatedUser(request);
    if (!user) {
      return createApiResponse(null, 'Unauthorized', 401);
    }

    const url = new URL(request.url);
    const batchId = url.searchParams.get('batch_id');
    const status = url.searchParams.get('status');
    const limit = parseInt(url.searchParams.get('limit') || '20');

    const { createSupabaseClient } = await import('@/lib/supabase/client');
    const supabase = createSupabaseClient();
    
    let query = supabase
      .from('embedding_job_progress')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(limit);

    if (status) {
      query = query.eq('status', status);
    }

    const { data: jobs, error } = await query;

    if (error) {
      console.error('Error getting batch jobs:', error);
      return createApiResponse(null, 'Error retrieving batch jobs', 500);
    }

    return createApiResponse({
      jobs: jobs || [],
      total: jobs?.length || 0,
      filters: {
        batch_id,
        status,
        limit
      }
    });

  } catch (error) {
    console.error('Error getting batch status:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
}
