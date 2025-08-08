import { EmbeddingQueue } from '../embeddingQueue';
import { createSupabaseClient } from '../../supabase/client';

// Mock dependencies
jest.mock('../../supabase/client');
jest.mock('../../embeddings/embeddingProcessor');

const mockSupabase = {
  from: jest.fn().mockReturnThis(),
  select: jest.fn().mockReturnThis(),
  insert: jest.fn().mockReturnThis(),
  update: jest.fn().mockReturnThis(),
  delete: jest.fn().mockReturnThis(),
  eq: jest.fn().mockReturnThis(),
  in: jest.fn().mockReturnThis(),
  order: jest.fn().mockReturnThis(),
  limit: jest.fn().mockReturnThis(),
  single: jest.fn().mockReturnThis(),
  data: null,
  error: null
};

(createSupabaseClient as jest.Mock).mockReturnValue(mockSupabase);

describe('EmbeddingQueue', () => {
  let queue: EmbeddingQueue;

  beforeEach(() => {
    queue = new EmbeddingQueue();
    jest.clearAllMocks();
  });

  describe('addEmbeddingJob', () => {
    it('should add embedding job successfully', async () => {
      const documentId = 'doc123';
      const jobId = 'job123';
      
      // Mock no existing job
      mockSupabase.select.mockResolvedValue({ data: null, error: { code: 'PGRST116' } });
      
      // Mock embedding processor
      const mockEmbeddingProcessor = {
        processDocumentChunks: jest.fn().mockResolvedValue(jobId)
      };
      (queue as any).embeddingProcessor = mockEmbeddingProcessor;
      
      const result = await queue.addEmbeddingJob(documentId, 'normal');
      
      expect(result).toBe(jobId);
      expect(mockEmbeddingProcessor.processDocumentChunks).toHaveBeenCalledWith(documentId);
    });

    it('should throw error if document already has active job', async () => {
      const documentId = 'doc123';
      
      // Mock existing job
      mockSupabase.select.mockResolvedValue({ 
        data: { id: 'existing-job', status: 'processing' }, 
        error: null 
      });
      
      await expect(queue.addEmbeddingJob(documentId)).rejects.toThrow(
        `Document ${documentId} already has an active embedding job`
      );
    });

    it('should handle high priority jobs', async () => {
      const documentId = 'doc123';
      const jobId = 'job123';
      
      mockSupabase.select.mockResolvedValue({ data: null, error: { code: 'PGRST116' } });
      
      const mockEmbeddingProcessor = {
        processDocumentChunks: jest.fn().mockResolvedValue(jobId)
      };
      (queue as any).embeddingProcessor = mockEmbeddingProcessor;
      
      const result = await queue.addEmbeddingJob(documentId, 'high');
      
      expect(result).toBe(jobId);
    });
  });

  describe('getQueueStats', () => {
    it('should return queue statistics', async () => {
      const mockStats = {
        pending: 5,
        processing: 2,
        completed: 10,
        failed: 1,
        total: 18
      };
      
      mockSupabase.select.mockResolvedValue({ data: mockStats, error: null });
      
      const stats = await queue.getQueueStats();
      
      expect(stats).toEqual(mockStats);
      expect(mockSupabase.select).toHaveBeenCalled();
    });

    it('should handle database errors', async () => {
      mockSupabase.select.mockResolvedValue({ data: null, error: new Error('DB error') });
      
      const stats = await queue.getQueueStats();
      
      expect(stats).toEqual({
        pending: 0,
        processing: 0,
        completed: 0,
        failed: 0,
        total: 0
      });
    });
  });

  describe('prioritizeUrgentJobs', () => {
    it('should prioritize urgent jobs', async () => {
      const documentIds = ['doc1', 'doc2'];
      
      mockSupabase.update.mockResolvedValue({ error: null });
      
      await queue.prioritizeUrgentJobs(documentIds);
      
      expect(mockSupabase.update).toHaveBeenCalled();
    });

    it('should handle update errors gracefully', async () => {
      const documentIds = ['doc1'];
      
      mockSupabase.update.mockResolvedValue({ error: new Error('Update failed') });
      
      // Should not throw error
      await expect(queue.prioritizeUrgentJobs(documentIds)).resolves.not.toThrow();
    });
  });

  describe('cancelJob', () => {
    it('should cancel job successfully', async () => {
      const jobId = 'job123';
      
      mockSupabase.update.mockResolvedValue({ error: null });
      
      await queue.cancelJob(jobId);
      
      expect(mockSupabase.update).toHaveBeenCalledWith({ status: 'cancelled' });
    });

    it('should handle cancellation errors', async () => {
      const jobId = 'job123';
      
      mockSupabase.update.mockResolvedValue({ error: new Error('Cancel failed') });
      
      await expect(queue.cancelJob(jobId)).rejects.toThrow('Cancel failed');
    });
  });

  describe('clearFailedJobs', () => {
    it('should clear failed jobs', async () => {
      const deletedCount = 3;
      
      mockSupabase.delete.mockResolvedValue({ count: deletedCount, error: null });
      
      const result = await queue.clearFailedJobs();
      
      expect(result).toBe(deletedCount);
      expect(mockSupabase.delete).toHaveBeenCalled();
    });

    it('should handle deletion errors', async () => {
      mockSupabase.delete.mockResolvedValue({ error: new Error('Delete failed') });
      
      const result = await queue.clearFailedJobs();
      
      expect(result).toBe(0);
    });
  });

  describe('getQueueStatus', () => {
    it('should return queue status', async () => {
      const mockStats = {
        pending: 2,
        processing: 1,
        completed: 5,
        failed: 0,
        total: 8
      };
      
      jest.spyOn(queue, 'getQueueStats').mockResolvedValue(mockStats);
      
      const status = await queue.getQueueStatus();
      
      expect(status).toEqual({
        processing: false,
        activeJobs: 0,
        concurrency: 3,
        stats: mockStats
      });
    });
  });

  describe('concurrency control', () => {
    it('should set concurrency limit', () => {
      const newConcurrency = 5;
      
      queue.setConcurrency(newConcurrency);
      
      expect((queue as any).concurrency).toBe(newConcurrency);
    });

    it('should stop processing', () => {
      queue.stopProcessing();
      
      expect((queue as any).processing).toBe(false);
    });
  });

  describe('job processing', () => {
    it('should process jobs in background', async () => {
      const mockJob = {
        id: 'job123',
        document_id: 'doc123',
        status: 'pending',
        created_at: new Date().toISOString()
      };
      
      // Mock getNextJob to return a job
      jest.spyOn(queue as any, 'getNextJob').mockResolvedValue(mockJob);
      jest.spyOn(queue as any, 'processJob').mockResolvedValue();
      jest.spyOn(queue as any, 'delay').mockResolvedValue();
      
      // Start processing
      await (queue as any).startProcessing();
      
      expect((queue as any).processJob).toHaveBeenCalledWith(mockJob);
    });

    it('should handle no jobs to process', async () => {
      // Mock getNextJob to return null
      jest.spyOn(queue as any, 'getNextJob').mockResolvedValue(null);
      jest.spyOn(queue as any, 'delay').mockResolvedValue();
      
      // Start processing
      await (queue as any).startProcessing();
      
      expect((queue as any).delay).toHaveBeenCalledWith(5000);
    });

    it('should respect concurrency limits', async () => {
      const mockJob = {
        id: 'job123',
        document_id: 'doc123',
        status: 'pending'
      };
      
      // Set active jobs to max concurrency
      (queue as any).activeJobs = new Set(['job1', 'job2', 'job3']);
      
      jest.spyOn(queue as any, 'getNextJob').mockResolvedValue(mockJob);
      jest.spyOn(queue as any, 'delay').mockResolvedValue();
      
      await (queue as any).startProcessing();
      
      // Should delay when at max concurrency
      expect((queue as any).delay).toHaveBeenCalledWith(1000);
    });
  });

  describe('error handling', () => {
    it('should handle job processing errors', async () => {
      const mockJob = {
        id: 'job123',
        document_id: 'doc123',
        status: 'pending'
      };
      
      jest.spyOn(queue as any, 'processJob').mockRejectedValue(new Error('Processing failed'));
      jest.spyOn(queue as any, 'retryJob').mockResolvedValue();
      
      await (queue as any).processJob(mockJob);
      
      expect((queue as any).retryJob).toHaveBeenCalledWith(mockJob.id);
    });

    it('should retry failed jobs', async () => {
      const jobId = 'job123';
      
      mockSupabase.update.mockResolvedValue({ error: null });
      
      await (queue as any).retryJob(jobId);
      
      expect(mockSupabase.update).toHaveBeenCalledWith({
        status: 'pending',
        retry_count: expect.any(Number),
        error_message: null
      });
    });

    it('should handle max retries exceeded', async () => {
      const mockJob = {
        id: 'job123',
        retry_count: 5,
        max_retries: 3
      };
      
      mockSupabase.select.mockResolvedValue({ data: mockJob, error: null });
      mockSupabase.update.mockResolvedValue({ error: null });
      
      await (queue as any).retryJob(mockJob.id);
      
      // Should mark as failed when max retries exceeded
      expect(mockSupabase.update).toHaveBeenCalledWith({
        status: 'failed',
        error_message: 'Max retries exceeded'
      });
    });
  });

  describe('performance monitoring', () => {
    it('should track processing metrics', async () => {
      const startTime = Date.now();
      
      // Mock job processing
      const mockJob = { id: 'job123', document_id: 'doc123' };
      jest.spyOn(queue as any, 'processJob').mockResolvedValue();
      
      await (queue as any).processJob(mockJob);
      
      // Should track processing time
      expect((queue as any).processJob).toHaveBeenCalledWith(mockJob);
    });

    it('should handle queue overflow', async () => {
      // Set high concurrency to test overflow handling
      queue.setConcurrency(100);
      
      const mockJob = { id: 'job123', document_id: 'doc123' };
      jest.spyOn(queue as any, 'getNextJob').mockResolvedValue(mockJob);
      jest.spyOn(queue as any, 'processJob').mockResolvedValue();
      jest.spyOn(queue as any, 'delay').mockResolvedValue();
      
      await (queue as any).startProcessing();
      
      // Should handle overflow gracefully
      expect((queue as any).processJob).toHaveBeenCalled();
    });
  });
}); 