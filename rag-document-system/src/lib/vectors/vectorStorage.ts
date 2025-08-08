import { createSupabaseClient } from '../supabase/client';
import { Embedding, SearchResult, VectorStats, EmbeddingCache } from '../types';

export class VectorStorageService {
  private supabase: ReturnType<typeof createSupabaseClient>;

  constructor() {
    this.supabase = createSupabaseClient();
  }

  /**
   * Store an embedding
   * @param embedding
   */
  async storeEmbedding(embedding: Embedding): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('embeddings')
        .insert(embedding);
      if (error) throw error;
    } catch (error) {
      console.error('Error storing embedding:', error);
    }
  }

  /**
   * Batch store embeddings
   * @param embeddings
   */
  async batchStoreEmbeddings(embeddings: Embedding[]): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('embeddings')
        .insert(embeddings, { returning: 'minimal' });
      if (error) throw error;
    } catch (error) {
      console.error('Error batch storing embeddings:', error);
    }
  }

  /**
   * Delete embeddings
   * @param chunkIds
   */
  async deleteEmbeddings(chunkIds: string[]): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('embeddings')
        .delete()
        .in('chunk_id', chunkIds);
      if (error) throw error;
    } catch (error) {
      console.error('Error deleting embeddings:', error);
    }
  }

  /**
   * Get vector statistics
   */
  async getEmbeddingStats(): Promise<VectorStats> {
    try {
      const { data, error } = await this.supabase
        .rpc('get_vector_stats')
        .single();
      if (error) throw error;
      return data as VectorStats;
    } catch (error) {
      console.error('Error getting vector stats:', error);
      return {
        total_embeddings: 0,
        total_documents: 0,
        avg_chunks_per_document: 0,
        cache_hit_rate: 0,
        storage_size_mb: 0,
      };
    }
  }

  /**
   * Get embedding from cache
   */
  async getFromCache(contentHash: string): Promise<EmbeddingCache | null> {
    try {
      const { data, error } = await this.supabase
        .from('embedding_cache')
        .select('*')
        .eq('content_hash', contentHash)
        .single();

      if (error || !data) return null;

      return data as EmbeddingCache;
    } catch (error) {
      console.error('Error fetching from cache:', error);
      return null;
    }
  }

  /**
   * Cache an embedding
   */
  async cacheEmbedding(embeddingCache: EmbeddingCache): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('embedding_cache')
        .upsert(embeddingCache, { onConflict: 'content_hash' });
      if (error) throw error;
    } catch (error) {
      console.error('Error caching embedding:', error);
    }
  }

  /**
   * Cleanup old cache
   */
  async cleanupExpiredCache(): Promise<number> {
    try {
      const { data, error } = await this.supabase
        .rpc('cleanup_expired_cache');
      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error cleaning up cache:', error);
      return 0;
    }
  }
}


// Export a singleton
export const vectorStorageService = new VectorStorageService();
export default vectorStorageService;

