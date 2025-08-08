import { NextRequest, NextResponse } from 'next/server';
import { vectorStorageService } from '@/lib/vectors/vectorStorage';
import { createApiResponse } from '@/lib/utils/response';
import { getAuthenticatedUser } from '@/lib/auth/utils';
import { createSupabaseClient } from '@/lib/supabase/client';

export async function DELETE(
  request: NextRequest,
  { params }: { params: { documentId: string } }
) {
  try {
    // Authenticate user
    const user = await getAuthenticatedUser(request);
    if (!user) {
      return createApiResponse(null, 'Unauthorized', 401);
    }

    const { documentId } = params;
    
    // Validate document exists and belongs to user
    const supabase = createSupabaseClient();
    
    const { data: document, error: docError } = await supabase
      .from('documents')
      .select('id, title, user_id')
      .eq('id', documentId)
      .eq('user_id', user.id)
      .single();

    if (docError || !document) {
      return createApiResponse(null, 'Document not found or access denied', 404);
    }

    // Get all chunk IDs for the document
    const { data: chunks, error: chunksError } = await supabase
      .from('document_chunks')
      .select('id')
      .eq('document_id', documentId);

    if (chunksError) {
      console.error('Error fetching document chunks:', chunksError);
      return createApiResponse(null, 'Error fetching document chunks', 500);
    }

    if (!chunks || chunks.length === 0) {
      return createApiResponse({
        document_id: documentId,
        deleted_embeddings: 0,
        message: 'No embeddings found for document'
      });
    }

    const chunkIds = chunks.map(chunk => chunk.id);

    // Delete embeddings for all chunks
    const { error: deleteError } = await supabase
      .from('embeddings')
      .delete()
      .in('chunk_id', chunkIds);

    if (deleteError) {
      console.error('Error deleting embeddings:', deleteError);
      return createApiResponse(null, 'Error deleting embeddings', 500);
    }

    // Also delete any related embedding jobs
    const { error: jobDeleteError } = await supabase
      .from('embedding_jobs')
      .delete()
      .eq('document_id', documentId);

    if (jobDeleteError) {
      console.error('Error deleting embedding jobs:', jobDeleteError);
      // Don't fail the request for job deletion errors
    }

    return createApiResponse({
      document_id: documentId,
      document_title: document.title,
      deleted_embeddings: chunks.length,
      deleted_chunks: chunks.length,
      message: 'All embeddings deleted successfully'
    });

  } catch (error) {
    console.error('Error deleting document embeddings:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { documentId: string } }
) {
  try {
    // Authenticate user
    const user = await getAuthenticatedUser(request);
    if (!user) {
      return createApiResponse(null, 'Unauthorized', 401);
    }

    const { documentId } = params;
    
    // Get embedding statistics for document
    const supabase = createSupabaseClient();
    
    const { data: embeddings, error } = await supabase
      .from('embeddings')
      .select(`
        id,
        chunk_id,
        model_version,
        processing_time_ms,
        status,
        created_at,
        document_chunks!inner (
          id,
          chunk_index,
          document_id,
          documents!inner (
            id,
            title,
            user_id
          )
        )
      `)
      .eq('document_chunks.document_id', documentId)
      .eq('document_chunks.documents.user_id', user.id);

    if (error) {
      console.error('Error fetching document embeddings:', error);
      return createApiResponse(null, 'Error fetching embeddings', 500);
    }

    // Calculate statistics
    const totalEmbeddings = embeddings?.length || 0;
    const completedEmbeddings = embeddings?.filter(e => e.status === 'completed').length || 0;
    const failedEmbeddings = embeddings?.filter(e => e.status === 'failed').length || 0;
    const pendingEmbeddings = embeddings?.filter(e => e.status === 'pending').length || 0;

    const avgProcessingTime = embeddings && embeddings.length > 0
      ? embeddings.reduce((sum, e) => sum + (e.processing_time_ms || 0), 0) / embeddings.length
      : 0;

    const modelVersions = embeddings
      ? [...new Set(embeddings.map(e => e.model_version).filter(Boolean))]
      : [];

    return createApiResponse({
      document_id: documentId,
      total_embeddings: totalEmbeddings,
      completed_embeddings: completedEmbeddings,
      failed_embeddings: failedEmbeddings,
      pending_embeddings: pendingEmbeddings,
      completion_rate: totalEmbeddings > 0 ? (completedEmbeddings / totalEmbeddings) * 100 : 0,
      avg_processing_time_ms: Math.round(avgProcessingTime),
      model_versions: modelVersions,
      last_updated: embeddings && embeddings.length > 0
        ? Math.max(...embeddings.map(e => new Date(e.created_at).getTime()))
        : null
    });

  } catch (error) {
    console.error('Error getting document embedding stats:', error);
    return createApiResponse(
      null,
      error instanceof Error ? error.message : 'Internal server error',
      500
    );
  }
} 