import { NextRequest, NextResponse } from 'next/server';
import { embeddingProcessor } from '@/lib/embeddings/embeddingProcessor';
import { embeddingQueue } from '@/lib/queue/embeddingQueue';
import { createApiResponse } from '@/lib/utils/response';
import { getAuthenticatedUser } from '@/lib/auth/utils';

export async function POST(
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
    
    // Get request body for options
    const body = await request.json().catch(() => ({}));
    const { priority = 'normal', force_reprocess = false } = body;

    // Validate document exists and belongs to user
    const { createSupabaseClient } = await import('@/lib/supabase/client');
    const supabase = createSupabaseClient();
    
    const { data: document, error: docError } = await supabase
      .from('documents')
      .select('id, title, processing_status')
      .eq('id', documentId)
      .eq('user_id', user.id)
      .single();

    if (docError || !document) {
      return createApiResponse(null, 'Document not found', 404);
    }

    // Check if document processing is complete
    if (document.processing_status !== 'completed') {
      return createApiResponse(null, 'Document must be processed before generating embeddings', 400);
    }

    // Check if embeddings already exist (unless force reprocess)
    if (!force_reprocess) {
      const { data: existingEmbeddings, error: embeddingError } = await supabase
        .from('embeddings')
        .select('id')
        .eq('document_id', documentId)
        .limit(1);

      if (embeddingError) {
        console.error('Error checking existing embeddings:', embeddingError);
      } else if (existingEmbeddings && existingEmbeddings.length > 0) {
        return createApiResponse(null, 'Embeddings already exist for this document', 400);
      }
    }

    // Add to embedding queue
    const jobId = await embeddingQueue.addEmbeddingJob(documentId, priority);
    
    // Get job status for response
    const jobStatus = await embeddingProcessor.getEmbeddingJobStatus(jobId);
    
    return createApiResponse({
      job_id: jobId,
      document_id: documentId,
      document_title: document.title,
      status: jobStatus?.status || 'pending',
      estimated_completion: jobStatus?.estimated_completion,
      message: 'Embedding generation started'
    });

  } catch (error) {
    console.error('Error generating embeddings:', error);
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
    
    // Get embedding jobs for document
    const jobs = await embeddingProcessor.getDocumentEmbeddingJobs(documentId);
    
    return createApiResponse({
      document_id: documentId,
      jobs: jobs
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
