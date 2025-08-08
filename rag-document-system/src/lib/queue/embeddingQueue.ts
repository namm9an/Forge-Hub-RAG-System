import { createSupabaseClient } from '../supabase/client';
import { embeddingProcessor } from '../embeddings/embeddingProcessor';
import { EmbeddingJob, EmbeddingJobProgress } from '../types';

export interface QueueStats {
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  total: number;
}

export interface QueueJob {
  id: string;
  document_id: string;
  priority: 'low' | 'normal' | 'high';
  created_at: string;
  scheduled_at: string;
  retry_count: number;
  max_retries: number;
}

export class EmbeddingQueue {
  private supabase: ReturnType<typeof createSupabaseClient>;
  private processing: boolean = false;
  private concurrency: number = 3;
  private activeJobs: Set<string> = new Set();

  constructor() {
    this.supabase = createSupabaseClient();
  }

  /**
   * Add embedding job to queue
   */
  async addEmbeddingJob(
    documentId: string,
    priority: 'low' | 'normal' | 'high' = 'normal'
  ): Promise<string> {
    try {
      // Check if document already has pending/processing job
      const { data: existingJob } = await this.supabase
        .from('embedding_jobs')
        .select('id, status')
        .eq('document_id', documentId)
        .in('status', ['pending', 'processing'])
        .single();

      if (existingJob) {
        throw new Error(`Document ${documentId} already has an active embedding job`);
      }

      // Create new job
      const jobId = await embeddingProcessor.processDocumentChunks(documentId);
      
      // Update job priority if not normal
      if (priority !== 'normal') {
        await this.updateJobPriority(jobId, priority);
      }

      // Start processing if not already running
      if (!this.processing) {
        this.startProcessing();
      }

      return jobId;

    } catch (error) {
      console.error('Failed to add embedding job:', error);
      throw error;
    }
  }

  /**
   * Start processing queue
   */
  private async startProcessing(): Promise<void> {
    if (this.processing) {
      return;
    }

    this.processing = true;
    console.log('Starting embedding queue processing');

    while (this.processing) {
      try {
        // Check if we can process more jobs
        if (this.activeJobs.size >= this.concurrency) {
          await this.delay(1000);
          continue;
        }

        // Get next job to process
        const job = await this.getNextJob();
        if (!job) {
          await this.delay(5000); // Wait 5 seconds before checking again
          continue;
        }

        // Process job
        this.processJob(job);

      } catch (error) {
        console.error('Error in queue processing:', error);
        await this.delay(5000);
      }
    }
  }

  /**
   * Get next job from queue
   */
  private async getNextJob(): Promise<EmbeddingJob | null> {
    const { data, error } = await this.supabase
      .from('embedding_jobs')
      .select('*')
      .eq('status', 'pending')
      .order('created_at', { ascending: true })
      .limit(1)
      .single();

    if (error || !data) {
      return null;
    }

    return data as EmbeddingJob;
  }

  /**
   * Process individual job
   */
  private async processJob(job: EmbeddingJob): Promise<void> {
    const jobId = job.id;
    
    try {
      this.activeJobs.add(jobId);
      console.log(`Processing job ${jobId} for document ${job.document_id}`);

      // Update job status to processing
      await this.supabase
        .from('embedding_jobs')
        .update({
          status: 'processing',
          started_at: new Date().toISOString()
        })
        .eq('id', jobId);

      // Get document chunks
      const { data: chunks, error: chunksError } = await this.supabase
        .from('document_chunks')
        .select('*')
        .eq('document_id', job.document_id)
        .order('chunk_index', { ascending: true });

      if (chunksError) throw chunksError;
      if (!chunks || chunks.length === 0) {
        throw new Error('No chunks found for document');
      }

      // Process embeddings
      await embeddingProcessor.processBatchEmbeddings(chunks, jobId);

      console.log(`Job ${jobId} completed successfully`);

    } catch (error) {
      console.error(`Job ${jobId} failed:`, error);
      
      // Update job status to failed
      await this.supabase
        .from('embedding_jobs')
        .update({
          status: 'failed',
          completed_at: new Date().toISOString(),
          error_message: error instanceof Error ? error.message : String(error),
          retry_count: job.retry_count + 1
        })
        .eq('id', jobId);

      // Retry if under max retries
      if (job.retry_count < 3) {
        await this.retryJob(jobId);
      }

    } finally {
      this.activeJobs.delete(jobId);
    }
  }

  /**
   * Retry failed job
   */
  private async retryJob(jobId: string): Promise<void> {
    const retryDelay = 60000; // 1 minute delay
    const scheduledAt = new Date(Date.now() + retryDelay);

    await this.supabase
      .from('embedding_jobs')
      .update({
        status: 'pending',
        started_at: null,
        completed_at: null,
        error_message: null
      })
      .eq('id', jobId);

    console.log(`Job ${jobId} scheduled for retry at ${scheduledAt.toISOString()}`);
  }

  /**
   * Update job priority
   */
  private async updateJobPriority(jobId: string, priority: 'low' | 'normal' | 'high'): Promise<void> {
    await this.supabase
      .from('embedding_jobs')
      .update({
        processing_metadata: {
          priority
        }
      })
      .eq('id', jobId);
  }

  /**
   * Get queue statistics
   */
  async getQueueStats(): Promise<QueueStats> {
    const { data, error } = await this.supabase
      .from('embedding_jobs')
      .select('status')
      .order('created_at', { ascending: false })
      .limit(1000); // Limit to recent jobs

    if (error) {
      console.error('Error getting queue stats:', error);
      return {
        pending: 0,
        processing: 0,
        completed: 0,
        failed: 0,
        total: 0
      };
    }

    const stats = data.reduce((acc, job) => {
      acc[job.status as keyof QueueStats]++;
      acc.total++;
      return acc;
    }, {
      pending: 0,
      processing: 0,
      completed: 0,
      failed: 0,
      total: 0
    });

    return stats;
  }

  /**
   * Prioritize urgent jobs
   */
  async prioritizeUrgentJobs(documentIds: string[]): Promise<void> {
    await this.supabase
      .from('embedding_jobs')
      .update({
        processing_metadata: {
          priority: 'high'
        }
      })
      .in('document_id', documentIds)
      .eq('status', 'pending');
  }

  /**
   * Cancel job
   */
  async cancelJob(jobId: string): Promise<void> {
    await this.supabase
      .from('embedding_jobs')
      .update({
        status: 'cancelled',
        completed_at: new Date().toISOString()
      })
      .eq('id', jobId);
  }

  /**
   * Clear failed jobs
   */
  async clearFailedJobs(): Promise<number> {
    const { data, error } = await this.supabase
      .from('embedding_jobs')
      .delete()
      .eq('status', 'failed')
      .select('id');

    if (error) {
      console.error('Error clearing failed jobs:', error);
      return 0;
    }

    return data?.length || 0;
  }

  /**
   * Get queue status
   */
  async getQueueStatus(): Promise<{
    processing: boolean;
    activeJobs: number;
    concurrency: number;
    stats: QueueStats;
  }> {
    const stats = await this.getQueueStats();
    
    return {
      processing: this.processing,
      activeJobs: this.activeJobs.size,
      concurrency: this.concurrency,
      stats
    };
  }

  /**
   * Stop queue processing
   */
  stopProcessing(): void {
    this.processing = false;
    console.log('Stopping embedding queue processing');
  }

  /**
   * Update concurrency
   */
  setConcurrency(concurrency: number): void {
    this.concurrency = Math.max(1, Math.min(10, concurrency));
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Export singleton instance
export const embeddingQueue = new EmbeddingQueue();
export default embeddingQueue;
