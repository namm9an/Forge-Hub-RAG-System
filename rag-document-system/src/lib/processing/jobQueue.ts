import { supabaseAdmin } from '@/lib/supabase/server';
import { parseDocumentFromBuffer } from './documentParser';
import { chunkText } from './textChunker';
import type { DocumentChunk } from './textChunker';

export interface ProcessingJob {
  id: string;
  document_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  retry_count: number;
  created_at: string;
  updated_at: string;
}

export interface ProcessingResult {
  success: boolean;
  chunkCount?: number;
  error?: string;
}

const MAX_RETRIES = 3;
const PROCESSING_TIMEOUT = 5 * 60 * 1000; // 5 minutes

/**
 * Add a new processing job to the queue
 */
export async function addProcessingJob(documentId: string): Promise<void> {
  try {
    const { error } = await supabaseAdmin
      .from('processing_jobs')
      .insert({
        document_id: documentId,
        status: 'pending',
        retry_count: 0,
      });

    if (error) {
      throw new Error(`Failed to create processing job: ${error.message}`);
    }

    // In a production environment, this would trigger a background worker
    // For now, we'll process it immediately in a non-blocking way
    setImmediate(() => {
      processNextJob().catch(console.error);
    });
  } catch (error) {
    console.error('Error adding processing job:', error);
    throw error;
  }
}

/**
 * Process the next pending job in the queue
 */
export async function processNextJob(): Promise<void> {
  try {
    // Get the next pending job
    const { data: job, error: fetchError } = await supabaseAdmin
      .from('processing_jobs')
      .select('*')
      .eq('status', 'pending')
      .order('created_at', { ascending: true })
      .limit(1)
      .single();

    if (fetchError || !job) {
      // No pending jobs
      return;
    }

    // Mark job as processing
    await updateJobStatus(job.id, 'processing', { started_at: new Date().toISOString() });

    // Process the document
    const result = await processDocument(job.document_id);

    if (result.success) {
      // Mark job as completed
      await updateJobStatus(job.id, 'completed', {
        completed_at: new Date().toISOString(),
      });

      // Update document status
      await supabaseAdmin
        .from('documents')
        .update({
          processing_status: 'completed',
          chunk_count: result.chunkCount || 0,
        })
        .eq('id', job.document_id);
    } else {
      // Handle failure
      await handleJobFailure(job, result.error || 'Unknown error');
    }

    // Process next job if any
    setImmediate(() => {
      processNextJob().catch(console.error);
    });
  } catch (error) {
    console.error('Error processing job:', error);
  }
}

/**
 * Update job status in the database
 */
export async function updateJobStatus(
  jobId: string,
  status: ProcessingJob['status'],
  additionalData: Partial<ProcessingJob> = {}
): Promise<void> {
  const { error } = await supabaseAdmin
    .from('processing_jobs')
    .update({
      status,
      ...additionalData,
      updated_at: new Date().toISOString(),
    })
    .eq('id', jobId);

  if (error) {
    console.error('Error updating job status:', error);
    throw error;
  }
}

/**
 * Process a document: parse and chunk it
 */
async function processDocument(documentId: string): Promise<ProcessingResult> {
  try {
    // Fetch document details
    const { data: document, error: docError } = await supabaseAdmin
      .from('documents')
      .select('*')
      .eq('id', documentId)
      .single();

    if (docError || !document) {
      throw new Error('Document not found');
    }

    // Download file from storage
    const { data: fileData, error: downloadError } = await supabaseAdmin
      .storage
      .from('documents')
      .download(document.upload_path);

    if (downloadError || !fileData) {
      throw new Error(`Failed to download file: ${downloadError?.message}`);
    }

    // Convert blob to buffer
    const buffer = Buffer.from(await fileData.arrayBuffer());

    // Parse document
    const parsedDocument = await parseDocumentFromBuffer(
      buffer,
      document.file_type,
      document.file_name
    );

    // Extract first 500 characters for preview
    const contentPreview = parsedDocument.text.substring(0, 500).trim();

    // Update document with preview and metadata
    await supabaseAdmin
      .from('documents')
      .update({
        content_preview: contentPreview,
        file_metadata: parsedDocument.metadata,
      })
      .eq('id', documentId);

    // Chunk the text
    const chunks = chunkText(parsedDocument.text, {
      pageCount: parsedDocument.metadata.pageCount,
    });

    // Prepare chunks for database insertion
    const chunksToInsert = chunks.map((chunk) => ({
      document_id: documentId,
      chunk_index: chunk.chunk_index,
      content: chunk.content,
      metadata: chunk.metadata,
    }));

    // Insert chunks in batches
    const batchSize = 100;
    for (let i = 0; i < chunksToInsert.length; i += batchSize) {
      const batch = chunksToInsert.slice(i, i + batchSize);
      const { error: chunkError } = await supabaseAdmin
        .from('document_chunks')
        .insert(batch);

      if (chunkError) {
        throw new Error(`Failed to insert chunks: ${chunkError.message}`);
      }
    }

    return {
      success: true,
      chunkCount: chunks.length,
    };
  } catch (error) {
    console.error('Document processing error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Handle job failure with retry logic
 */
async function handleJobFailure(job: ProcessingJob, errorMessage: string): Promise<void> {
  const retryCount = job.retry_count + 1;

  if (retryCount >= MAX_RETRIES) {
    // Max retries reached, mark as failed
    await updateJobStatus(job.id, 'failed', {
      error_message: errorMessage,
      completed_at: new Date().toISOString(),
    });

    // Update document status
    await supabaseAdmin
      .from('documents')
      .update({
        processing_status: 'failed',
        processing_error: errorMessage,
      })
      .eq('id', job.document_id);
  } else {
    // Retry the job
    await updateJobStatus(job.id, 'pending', {
      retry_count: retryCount,
      error_message: errorMessage,
    });

    // Schedule retry with exponential backoff
    const delay = Math.pow(2, retryCount) * 1000; // 2s, 4s, 8s
    setTimeout(() => {
      processNextJob().catch(console.error);
    }, delay);
  }
}

/**
 * Get processing queue status
 */
export async function getQueueStatus(): Promise<{
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  avgProcessingTime?: number;
}> {
  try {
    // Get counts by status
    const { data: statusCounts, error: countError } = await supabaseAdmin
      .from('processing_jobs')
      .select('status')
      .order('created_at', { ascending: false })
      .limit(1000);

    if (countError) {
      throw countError;
    }

    const counts = {
      pending: 0,
      processing: 0,
      completed: 0,
      failed: 0,
    };

    statusCounts?.forEach((job) => {
      counts[job.status as keyof typeof counts]++;
    });

    // Calculate average processing time for completed jobs
    const { data: completedJobs, error: timeError } = await supabaseAdmin
      .from('processing_jobs')
      .select('started_at, completed_at')
      .eq('status', 'completed')
      .not('started_at', 'is', null)
      .not('completed_at', 'is', null)
      .order('completed_at', { ascending: false })
      .limit(100);

    let avgProcessingTime: number | undefined;
    if (completedJobs && completedJobs.length > 0) {
      const totalTime = completedJobs.reduce((sum, job) => {
        const start = new Date(job.started_at!).getTime();
        const end = new Date(job.completed_at!).getTime();
        return sum + (end - start);
      }, 0);
      avgProcessingTime = Math.round(totalTime / completedJobs.length / 1000); // in seconds
    }

    return {
      ...counts,
      avgProcessingTime,
    };
  } catch (error) {
    console.error('Error getting queue status:', error);
    throw error;
  }
}

/**
 * Clean up old completed jobs (maintenance task)
 */
export async function cleanupOldJobs(daysToKeep: number = 7): Promise<number> {
  try {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);

    const { data, error } = await supabaseAdmin
      .from('processing_jobs')
      .delete()
      .in('status', ['completed', 'failed'])
      .lt('completed_at', cutoffDate.toISOString())
      .select();

    if (error) {
      throw error;
    }

    return data?.length || 0;
  } catch (error) {
    console.error('Error cleaning up old jobs:', error);
    throw error;
  }
}
