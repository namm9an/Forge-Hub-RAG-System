import { GeminiEmbeddingService } from '../geminiEmbeddings';
import { createSupabaseClient } from '../../supabase/client';

// Mock dependencies
jest.mock('../../supabase/client');
jest.mock('../../gemini/client');

const mockSupabase = {
  from: jest.fn().mockReturnThis(),
  select: jest.fn().mockReturnThis(),
  insert: jest.fn().mockReturnThis(),
  eq: jest.fn().mockReturnThis(),
  single: jest.fn().mockReturnThis(),
  rpc: jest.fn().mockReturnThis(),
  data: null,
  error: null
};

(createSupabaseClient as jest.Mock).mockReturnValue(mockSupabase);

describe('GeminiEmbeddingService', () => {
  let service: GeminiEmbeddingService;

  beforeEach(() => {
    service = new GeminiEmbeddingService();
    jest.clearAllMocks();
  });

  describe('generateEmbedding', () => {
    it('should generate embedding for valid text', async () => {
      const mockEmbedding = new Array(768).fill(0.1);
      const mockResponse = {
        embedding: mockEmbedding,
        model_version: 'text-embedding-004',
        processing_time_ms: 100,
        dimensions: 768,
        cached: false,
        metadata: {}
      };

      // Mock the internal methods
      jest.spyOn(service as any, 'generateEmbeddingWithRetry').mockResolvedValue(mockEmbedding);
      jest.spyOn(service as any, 'preprocessText').mockReturnValue('processed text');
      jest.spyOn(service as any, 'checkRateLimit').mockResolvedValue();
      jest.spyOn(service as any, 'trackUsage').mockResolvedValue();

      const result = await service.generateEmbedding({
        text: 'test text',
        cache_result: true
      });

      expect(result).toEqual(mockResponse);
      expect(result.embedding).toHaveLength(768);
      expect(result.model_version).toBe('text-embedding-004');
    });

    it('should return cached embedding if available', async () => {
      const mockCachedEmbedding = {
        embedding: new Array(768).fill(0.2),
        model_version: 'text-embedding-004',
        usage_count: 1,
        created_at: new Date().toISOString()
      };

      jest.spyOn(service as any, 'getCachedEmbedding').mockResolvedValue(mockCachedEmbedding);
      jest.spyOn(service as any, 'preprocessText').mockReturnValue('processed text');

      const result = await service.generateEmbedding({
        text: 'cached text',
        cache_result: true
      });

      expect(result.cached).toBe(true);
      expect(result.embedding).toEqual(mockCachedEmbedding.embedding);
    });

    it('should throw error for empty text after preprocessing', async () => {
      jest.spyOn(service as any, 'preprocessText').mockReturnValue('');

      await expect(service.generateEmbedding({
        text: 'empty text',
        cache_result: true
      })).rejects.toThrow('Text cannot be empty after preprocessing');
    });

    it('should handle rate limit errors with retry', async () => {
      const mockEmbedding = new Array(768).fill(0.1);
      
      jest.spyOn(service as any, 'preprocessText').mockReturnValue('processed text');
      jest.spyOn(service as any, 'checkRateLimit').mockResolvedValue();
      jest.spyOn(service as any, 'generateEmbeddingWithRetry')
        .mockRejectedValueOnce(new Error('Rate limit exceeded'))
        .mockResolvedValue(mockEmbedding);
      jest.spyOn(service as any, 'trackUsage').mockResolvedValue();

      const result = await service.generateEmbedding({
        text: 'test text',
        cache_result: true
      });

      expect(result.embedding).toEqual(mockEmbedding);
    });
  });

  describe('generateBatchEmbeddings', () => {
    it('should generate embeddings for batch of texts', async () => {
      const texts = ['text1', 'text2', 'text3'];
      const mockEmbeddings = texts.map(() => new Array(768).fill(0.1));

      jest.spyOn(service as any, 'preprocessText').mockImplementation(text => text);
      jest.spyOn(service as any, 'checkRateLimit').mockResolvedValue();
      jest.spyOn(service as any, 'generateEmbeddingWithRetry')
        .mockResolvedValueOnce(mockEmbeddings[0])
        .mockResolvedValueOnce(mockEmbeddings[1])
        .mockResolvedValueOnce(mockEmbeddings[2]);
      jest.spyOn(service as any, 'trackUsage').mockResolvedValue();

      const results = await service.generateBatchEmbeddings(texts);

      expect(results).toHaveLength(3);
      results.forEach((result, index) => {
        expect(result.embedding).toEqual(mockEmbeddings[index]);
        expect(result.embedding).toHaveLength(768);
      });
    });

    it('should handle empty batch', async () => {
      const results = await service.generateBatchEmbeddings([]);
      expect(results).toEqual([]);
    });

    it('should respect batch size limits', async () => {
      const largeTexts = Array(25).fill('test text');
      
      jest.spyOn(service as any, 'preprocessText').mockImplementation(text => text);
      jest.spyOn(service as any, 'checkRateLimit').mockResolvedValue();
      jest.spyOn(service as any, 'generateEmbeddingWithRetry').mockResolvedValue(new Array(768).fill(0.1));
      jest.spyOn(service as any, 'trackUsage').mockResolvedValue();

      const results = await service.generateBatchEmbeddings(largeTexts);

      expect(results).toHaveLength(25);
    });
  });

  describe('validateEmbedding', () => {
    it('should validate correct embedding dimensions', () => {
      const validEmbedding = new Array(768).fill(0.1);
      expect(service.validateEmbedding(validEmbedding)).toBe(true);
    });

    it('should reject embedding with wrong dimensions', () => {
      const invalidEmbedding = new Array(512).fill(0.1);
      expect(service.validateEmbedding(invalidEmbedding)).toBe(false);
    });

    it('should reject empty embedding', () => {
      expect(service.validateEmbedding([])).toBe(false);
    });

    it('should reject embedding with invalid values', () => {
      const invalidEmbedding = new Array(768).fill(NaN);
      expect(service.validateEmbedding(invalidEmbedding)).toBe(false);
    });
  });

  describe('rate limiting', () => {
    it('should respect rate limits', async () => {
      const mockRateLimiter = {
        consume: jest.fn().mockResolvedValue({ remainingPoints: 5 })
      };
      (service as any).rateLimiter = mockRateLimiter;

      jest.spyOn(service as any, 'preprocessText').mockReturnValue('test');
      jest.spyOn(service as any, 'generateEmbeddingWithRetry').mockResolvedValue(new Array(768).fill(0.1));
      jest.spyOn(service as any, 'trackUsage').mockResolvedValue();

      await service.generateEmbedding({ text: 'test' });

      expect(mockRateLimiter.consume).toHaveBeenCalled();
    });

    it('should wait when rate limit exceeded', async () => {
      const mockRateLimiter = {
        consume: jest.fn().mockRejectedValue(new Error('Rate limit exceeded'))
      };
      (service as any).rateLimiter = mockRateLimiter;

      jest.spyOn(service as any, 'delay').mockResolvedValue();
      jest.spyOn(service as any, 'preprocessText').mockReturnValue('test');
      jest.spyOn(service as any, 'generateEmbeddingWithRetry').mockResolvedValue(new Array(768).fill(0.1));
      jest.spyOn(service as any, 'trackUsage').mockResolvedValue();

      await service.generateEmbedding({ text: 'test' });

      expect(service as any).toHaveProperty('delay');
    });
  });

  describe('configuration', () => {
    it('should return current configuration', () => {
      const config = service.getConfiguration();
      expect(config).toHaveProperty('batch_size');
      expect(config).toHaveProperty('rate_limit');
      expect(config).toHaveProperty('preprocessing');
    });

    it('should update configuration', () => {
      const newConfig = { batch_size: 20 };
      service.updateConfiguration(newConfig);
      
      const config = service.getConfiguration();
      expect(config.batch_size).toBe(20);
    });
  });

  describe('connection testing', () => {
    it('should test connection successfully', async () => {
      jest.spyOn(service as any, 'generateEmbeddingWithRetry').mockResolvedValue(new Array(768).fill(0.1));
      
      const result = await service.testConnection();
      expect(result).toBe(true);
    });

    it('should handle connection failure', async () => {
      jest.spyOn(service as any, 'generateEmbeddingWithRetry').mockRejectedValue(new Error('Connection failed'));
      
      const result = await service.testConnection();
      expect(result).toBe(false);
    });
  });
}); 