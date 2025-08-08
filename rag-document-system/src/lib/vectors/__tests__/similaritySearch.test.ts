import { SimilaritySearchService } from '../similaritySearch';
import { createSupabaseClient } from '../../supabase/client';

// Mock dependencies
jest.mock('../../supabase/client');
jest.mock('../../embeddings/geminiEmbeddings');
jest.mock('../../embeddings/textPreprocessor');

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

describe('SimilaritySearchService', () => {
  let service: SimilaritySearchService;

  beforeEach(() => {
    service = new SimilaritySearchService();
    jest.clearAllMocks();
  });

  describe('searchSimilarity', () => {
    it('should perform similarity search successfully', async () => {
      const mockQueryEmbedding = new Array(768).fill(0.1);
      const mockSearchResults = [
        {
          chunk_id: 'chunk1',
          document_id: 'doc1',
          chunk_index: 0,
          content: 'test content 1',
          chunk_metadata: {},
          chunk_created_at: new Date().toISOString(),
          document_title: 'Test Document',
          file_name: 'test.pdf',
          file_type: 'pdf',
          user_id: 'user1',
          document_created_at: new Date().toISOString(),
          document_updated_at: new Date().toISOString(),
          similarity: 0.85,
          model_version: 'text-embedding-004',
          processing_time_ms: 100
        }
      ];

      // Mock dependencies
      jest.spyOn(service as any, 'generateQueryEmbedding').mockResolvedValue(mockQueryEmbedding);
      mockSupabase.rpc.mockResolvedValue({ data: mockSearchResults, error: null });

      const result = await service.searchSimilarity({
        query: 'test query',
        similarity_threshold: 0.7,
        limit: 10
      });

      expect(result).toHaveLength(1);
      expect(result[0].similarity_score).toBe(0.85);
      expect(result[0].chunk.content).toBe('test content 1');
    });

    it('should handle empty search results', async () => {
      jest.spyOn(service as any, 'generateQueryEmbedding').mockResolvedValue(new Array(768).fill(0.1));
      mockSupabase.rpc.mockResolvedValue({ data: [], error: null });

      const result = await service.searchSimilarity({
        query: 'nonexistent query',
        similarity_threshold: 0.9,
        limit: 10
      });

      expect(result).toEqual([]);
    });

    it('should throw error for invalid query', async () => {
      await expect(service.searchSimilarity({
        query: '',
        similarity_threshold: 0.7,
        limit: 10
      })).rejects.toThrow('Query is empty or invalid after preprocessing');
    });

    it('should apply similarity threshold correctly', async () => {
      const mockSearchResults = [
        { similarity: 0.9, content: 'high similarity' },
        { similarity: 0.5, content: 'low similarity' }
      ];

      jest.spyOn(service as any, 'generateQueryEmbedding').mockResolvedValue(new Array(768).fill(0.1));
      mockSupabase.rpc.mockResolvedValue({ data: mockSearchResults, error: null });

      const result = await service.searchSimilarity({
        query: 'test query',
        similarity_threshold: 0.7,
        limit: 10
      });

      // Should only return results above threshold
      expect(result.every(r => r.similarity_score >= 0.7)).toBe(true);
    });
  });

  describe('calculateCosineSimilarity', () => {
    it('should calculate cosine similarity correctly', () => {
      const vectorA = [1, 0, 0];
      const vectorB = [1, 0, 0];
      
      const similarity = service.calculateCosineSimilarity(vectorA, vectorB);
      expect(similarity).toBe(1);
    });

    it('should calculate cosine similarity for orthogonal vectors', () => {
      const vectorA = [1, 0, 0];
      const vectorB = [0, 1, 0];
      
      const similarity = service.calculateCosineSimilarity(vectorA, vectorB);
      expect(similarity).toBe(0);
    });

    it('should handle zero vectors', () => {
      const vectorA = [0, 0, 0];
      const vectorB = [1, 1, 1];
      
      const similarity = service.calculateCosineSimilarity(vectorA, vectorB);
      expect(similarity).toBe(0);
    });

    it('should handle vectors with different magnitudes', () => {
      const vectorA = [1, 1, 1];
      const vectorB = [2, 2, 2];
      
      const similarity = service.calculateCosineSimilarity(vectorA, vectorB);
      expect(similarity).toBe(1);
    });
  });

  describe('generateHighlights', () => {
    it('should generate highlights for matching content', () => {
      const content = 'This is a test document with important information about machine learning.';
      const query = 'machine learning';
      
      const highlights = service.generateHighlights(content, query);
      
      expect(highlights).toContain('machine learning');
      expect(highlights.length).toBeGreaterThan(0);
    });

    it('should handle case insensitive matching', () => {
      const content = 'This document contains MACHINE LEARNING concepts.';
      const query = 'machine learning';
      
      const highlights = service.generateHighlights(content, query);
      
      expect(highlights.some(h => h.toLowerCase().includes('machine learning'))).toBe(true);
    });

    it('should return empty array for no matches', () => {
      const content = 'This document has no relevant content.';
      const query = 'nonexistent term';
      
      const highlights = service.generateHighlights(content, query);
      
      expect(highlights).toEqual([]);
    });
  });

  describe('rankSearchResults', () => {
    it('should rank results by similarity score', () => {
      const results = [
        { similarity_score: 0.5, content: 'low similarity' },
        { similarity_score: 0.9, content: 'high similarity' },
        { similarity_score: 0.7, content: 'medium similarity' }
      ];
      
      const ranked = service.rankSearchResults(results);
      
      expect(ranked[0].similarity_score).toBe(0.9);
      expect(ranked[1].similarity_score).toBe(0.7);
      expect(ranked[2].similarity_score).toBe(0.5);
    });

    it('should handle empty results', () => {
      const results = [];
      const ranked = service.rankSearchResults(results);
      expect(ranked).toEqual([]);
    });
  });

  describe('cacheSearchResults', () => {
    it('should cache search results', async () => {
      const query = 'test query';
      const results = [
        { similarity_score: 0.8, content: 'test result' }
      ];
      
      mockSupabase.insert.mockResolvedValue({ error: null });
      
      await service.cacheSearchResults(query, results);
      
      expect(mockSupabase.insert).toHaveBeenCalled();
    });

    it('should handle cache insertion errors', async () => {
      const query = 'test query';
      const results = [{ similarity_score: 0.8, content: 'test result' }];
      
      mockSupabase.insert.mockResolvedValue({ error: new Error('Cache error') });
      
      // Should not throw error
      await expect(service.cacheSearchResults(query, results)).resolves.not.toThrow();
    });
  });

  describe('getCachedSearchResults', () => {
    it('should retrieve cached results', async () => {
      const query = 'test query';
      const mockCachedResults = {
        query_text: query,
        search_results: [{ similarity_score: 0.8, content: 'cached result' }],
        similarity_threshold: 0.7,
        result_count: 1,
        created_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 3600000).toISOString()
      };
      
      mockSupabase.select.mockResolvedValue({ data: mockCachedResults, error: null });
      
      const result = await service.getCachedSearchResults(query);
      
      expect(result).toEqual(mockCachedResults.search_results);
    });

    it('should return null for expired cache', async () => {
      const query = 'test query';
      const mockCachedResults = {
        query_text: query,
        search_results: [{ similarity_score: 0.8, content: 'cached result' }],
        similarity_threshold: 0.7,
        result_count: 1,
        created_at: new Date().toISOString(),
        expires_at: new Date(Date.now() - 3600000).toISOString() // Expired
      };
      
      mockSupabase.select.mockResolvedValue({ data: mockCachedResults, error: null });
      
      const result = await service.getCachedSearchResults(query);
      
      expect(result).toBeNull();
    });

    it('should return null for no cached results', async () => {
      const query = 'test query';
      
      mockSupabase.select.mockResolvedValue({ data: null, error: null });
      
      const result = await service.getCachedSearchResults(query);
      
      expect(result).toBeNull();
    });
  });

  describe('performHybridSearch', () => {
    it('should perform hybrid search with semantic and keyword components', async () => {
      const mockSemanticResults = [
        { similarity_score: 0.8, content: 'semantic result' }
      ];
      const mockKeywordResults = [
        { similarity_score: 0.6, content: 'keyword result' }
      ];
      
      jest.spyOn(service, 'searchSimilarity').mockResolvedValue(mockSemanticResults);
      jest.spyOn(service as any, 'keywordSearch').mockResolvedValue(mockKeywordResults);
      
      const result = await service.performHybridSearch({
        query: 'test query',
        semantic_weight: 0.7,
        keyword_weight: 0.3,
        similarity_threshold: 0.7,
        limit: 10
      });
      
      expect(result.length).toBeGreaterThan(0);
      expect(service.searchSimilarity).toHaveBeenCalled();
    });

    it('should handle empty results from both searches', async () => {
      jest.spyOn(service, 'searchSimilarity').mockResolvedValue([]);
      jest.spyOn(service as any, 'keywordSearch').mockResolvedValue([]);
      
      const result = await service.performHybridSearch({
        query: 'nonexistent query',
        semantic_weight: 0.7,
        keyword_weight: 0.3,
        similarity_threshold: 0.9,
        limit: 10
      });
      
      expect(result).toEqual([]);
    });
  });

  describe('error handling', () => {
    it('should handle database errors gracefully', async () => {
      jest.spyOn(service as any, 'generateQueryEmbedding').mockResolvedValue(new Array(768).fill(0.1));
      mockSupabase.rpc.mockResolvedValue({ data: null, error: new Error('Database error') });
      
      await expect(service.searchSimilarity({
        query: 'test query',
        similarity_threshold: 0.7,
        limit: 10
      })).rejects.toThrow('Database error');
    });

    it('should handle embedding generation errors', async () => {
      jest.spyOn(service as any, 'generateQueryEmbedding').mockRejectedValue(new Error('Embedding error'));
      
      await expect(service.searchSimilarity({
        query: 'test query',
        similarity_threshold: 0.7,
        limit: 10
      })).rejects.toThrow('Embedding error');
    });
  });
}); 