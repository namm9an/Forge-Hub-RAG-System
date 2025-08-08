import { NextRequest, NextResponse } from 'next/server';
import { geminiEmbeddingService } from '@/lib/embeddings/geminiEmbeddings';
import { vectorStorageService } from '@/lib/vectors/vectorStorage';
import { createApiResponse } from '@/lib/utils/response';
import { getAuthenticatedUser } from '@/lib/auth/utils';
import { createSupabaseClient } from '@/lib/supabase/client';

export async function POST(
  request: NextRequest,
  { params }: { params: { chunkId: string } }
) {
  try {
    // Authenticate user
    const user = await getAuthenticatedUser(request);
    if (!user) {
      return createApiResponse(null, 'Unauthorized', 401);
    }

    const { chunkId } = params;
    
    // Get request body for options
    const body = await request.json().catch(() => ({}));
    const { model_version = 'text-embedding-004', force_reprocess = false } = body;

    // Validate chunk exists and belongs to user
    const supabase = createSupabaseClient();
    
    const { data: chunk, error: chunkError } = await supabase
      .from('document_chunks')
      .select(`
        id,
        content,
        chunk_index,
        metadata,
        document_id,
        documents!inner (
          id,
          title,
          user_id
        )
      `)
      .eq('id', chunkId)
      .eq('documents.user_id', user.id)
      .single();

    if (chunkError || !chunk) {
      return createApiResponse(null, 'Chunk not found or access denied', 404);
    }

    // Check if embedding already exists (unless force reprocess)
    if (!force_reprocess) {
      const { data: existingEmbedding, error: embeddingError } = await supabase
        .from('embeddings')
        .select('id, model_version')
        .eq('chunk_id', chunkId)
        .single();

      if (!embeddingError && existingEmbedding) {
        return createApiResponse(null, 'Embedding already exists for this chunk', 400);
      }
    }

    // Generate new embedding
    const startTime = Date.now();
    const embeddingResponse = await geminiEmbeddingService.generateEmbedding({
      text: chunk.content,
      model_version,
      cache_result: true
    });

    // Store the embedding
    const embedding = {
      chunk_id: chunkId,
      embedding: embeddingResponse.embedding,
      model_version: embeddingResponse.model_version,
      processing_time_ms: Date.now() - startTime,
      status: 'completed',
      embedding_metadata: {
        reprocessed: true,
        original_model: existingEmbedding?.model_version,
        reprocessed_at: new Date().toISOString()
      }
    };

    // Delete existing embedding if it exists
    if (force_reprocess) {
      await supabase
        .from('embeddings')
        .delete()
        .eq('chunk_id', chunkId);
    }

    // Store new embedding
    const { error: storeError } = await supabase
      .from('embeddings')
      .insert(embedding);

    if (storeError) {
      console.error('Error storing reprocessed embedding:', storeError);
      return createApiResponse(null, 'Failed to store embedding', 500);
    }

    return createApiResponse({
      chunk_id: chunkId,
      document_id: chunk.document_id,
      model_version: embeddingResponse.model_version,
      processing_time_ms: embedding.processing_time_ms,
      dimensions: embeddingResponse.embedding.length,
      message: 'Embedding reprocessed successfully'
    });

  } catch (error) {
    console.error('Error reprocessing embedding:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { chunkId: string } }
) {
  try {
    // Authenticate user
    const user = await getAuthenticatedUser(request);
    if (!user) {
      return createApiResponse(null, 'Unauthorized', 401);
    }

    const { chunkId } = params;
    
    // Get embedding status for chunk
    const supabase = createSupabaseClient();
    
    const { data: embedding, error } = await supabase
      .from('embeddings')
      .select(`
        id,
        chunk_id,
        model_version,
        processing_time_ms,
        status,
        embedding_metadata,
        created_at,
        updated_at,
        document_chunks!inner (
          id,
          content,
          chunk_index,
          document_id,
          documents!inner (
            id,
            title,
            user_id
          )
        )
      `)
      .eq('chunk_id', chunkId)
      .eq('document_chunks.documents.user_id', user.id)
      .single();

    if (error || !embedding) {
      return createApiResponse(null, 'Embedding not found or access denied', 404);
    }

    return createApiResponse({
      chunk_id: chunkId,
      document_id: embedding.document_chunks.document_id,
      model_version: embedding.model_version,
      status: embedding.status,
      processing_time_ms: embedding.processing_time_ms,
      created_at: embedding.created_at,
      updated_at: embedding.updated_at,
      metadata: embedding.embedding_metadata
    });

  } catch (error) {
    console.error('Error getting embedding status:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
} 