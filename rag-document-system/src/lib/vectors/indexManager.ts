import { createSupabaseClient } from '../supabase/client';
import { VectorIndexStats } from '../types';

export class IndexManager {
  private supabase: ReturnType<typeof createSupabaseClient>;

  constructor() {
    this.supabase = createSupabaseClient();
  }

  /**
   * Create vector index for embeddings
   */
  async createVectorIndex(): Promise<void> {
    try {
      // Create ivfflat indexes for different distance metrics
      const queries = [
        'CREATE INDEX IF NOT EXISTS embeddings_vector_cosine_idx ON public.embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)',
        'CREATE INDEX IF NOT EXISTS embeddings_vector_l2_idx ON public.embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)',
        'CREATE INDEX IF NOT EXISTS embeddings_vector_ip_idx ON public.embeddings USING ivfflat (embedding vector_ip_ops) WITH (lists = 100)'
      ];

      for (const query of queries) {
        const { error } = await this.supabase.rpc('execute_sql', { query });
        if (error) {
          console.error(`Failed to create index: ${query}`, error);
        }
      }

      console.log('Vector indexes created successfully');
    } catch (error) {
      console.error('Error creating vector indexes:', error);
      throw error;
    }
  }

  /**
   * Rebuild vector index for optimization
   */
  async rebuildVectorIndex(): Promise<void> {
    try {
      const { error } = await this.supabase.rpc('optimize_vector_indexes');
      if (error) throw error;
      
      console.log('Vector indexes rebuilt successfully');
    } catch (error) {
      console.error('Error rebuilding vector indexes:', error);
      throw error;
    }
  }

  /**
   * Optimize index performance
   */
  async optimizeIndexPerformance(): Promise<void> {
    try {
      // Update statistics
      await this.supabase.rpc('execute_sql', {
        query: 'ANALYZE public.embeddings, public.document_chunks, public.documents'
      });

      // Reindex if needed
      await this.rebuildVectorIndex();

      console.log('Index performance optimized');
    } catch (error) {
      console.error('Error optimizing index performance:', error);
      throw error;
    }
  }

  /**
   * Get index health statistics
   */
  async getIndexHealth(): Promise<VectorIndexStats[]> {
    try {
      const { data, error } = await this.supabase.rpc('get_index_stats');
      if (error) throw error;

      return data || [];
    } catch (error) {
      console.error('Error getting index health:', error);
      return [];
    }
  }

  /**
   * Monitor index usage
   */
  async monitorIndexUsage(): Promise<{
    total_vectors: number;
    index_size_mb: number;
    avg_query_time_ms: number;
    recent_queries: number;
  }> {
    try {
      const { data: stats, error } = await this.supabase.rpc('get_vector_stats');
      if (error) throw error;

      return {
        total_vectors: stats.total_embeddings,
        index_size_mb: stats.storage_size_mb,
        avg_query_time_ms: 0, // Would need query performance tracking
        recent_queries: 0 // Would need query tracking
      };
    } catch (error) {
      console.error('Error monitoring index usage:', error);
      return {
        total_vectors: 0,
        index_size_mb: 0,
        avg_query_time_ms: 0,
        recent_queries: 0
      };
    }
  }

  /**
   * Auto-optimize indexes based on usage
   */
  async autoOptimizeIndexes(): Promise<void> {
    try {
      const usage = await this.monitorIndexUsage();
      
      // Rebuild if we have significant number of vectors
      if (usage.total_vectors > 10000) {
        console.log('Auto-optimizing indexes due to high vector count');
        await this.optimizeIndexPerformance();
      }

      // Check if indexes need maintenance
      const health = await this.getIndexHealth();
      const unhealthyIndexes = health.filter(idx => idx.health_status !== 'healthy');
      
      if (unhealthyIndexes.length > 0) {
        console.log(`Rebuilding ${unhealthyIndexes.length} unhealthy indexes`);
        await this.rebuildVectorIndex();
      }

    } catch (error) {
      console.error('Error in auto-optimization:', error);
    }
  }

  /**
   * Drop and recreate indexes
   */
  async dropAndRecreateIndexes(): Promise<void> {
    try {
      // Drop existing indexes
      const dropQueries = [
        'DROP INDEX IF EXISTS embeddings_vector_cosine_idx',
        'DROP INDEX IF EXISTS embeddings_vector_l2_idx',
        'DROP INDEX IF EXISTS embeddings_vector_ip_idx'
      ];

      for (const query of dropQueries) {
        await this.supabase.rpc('execute_sql', { query });
      }

      // Recreate indexes
      await this.createVectorIndex();
      
      console.log('Indexes dropped and recreated successfully');
    } catch (error) {
      console.error('Error dropping and recreating indexes:', error);
      throw error;
    }
  }

  /**
   * Get index configuration recommendations
   */
  async getIndexRecommendations(): Promise<{
    recommended_lists: number;
    recommended_probes: number;
    reason: string;
  }> {
    try {
      const usage = await this.monitorIndexUsage();
      
      let recommendedLists = 100;
      let recommendedProbes = 10;
      let reason = 'Default configuration';

      if (usage.total_vectors > 100000) {
        recommendedLists = 200;
        recommendedProbes = 20;
        reason = 'High vector count detected';
      } else if (usage.total_vectors > 50000) {
        recommendedLists = 150;
        recommendedProbes = 15;
        reason = 'Medium vector count detected';
      }

      return {
        recommended_lists: recommendedLists,
        recommended_probes: recommendedProbes,
        reason
      };
    } catch (error) {
      console.error('Error getting index recommendations:', error);
      return {
        recommended_lists: 100,
        recommended_probes: 10,
        reason: 'Default due to error'
      };
    }
  }
}

// Export singleton instance
export const indexManager = new IndexManager();
export default indexManager;
