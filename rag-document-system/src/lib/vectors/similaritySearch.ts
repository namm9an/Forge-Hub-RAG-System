import { createSupabaseClient } from '../supabase/client';
import { SearchResult, SimilaritySearchRequest, SimilaritySearchResult } from '../types';
import { textPreprocessor } from '../embeddings/textPreprocessor';
import { geminiEmbeddingService } from '../embeddings/geminiEmbeddings';

export class SimilaritySearchService {
  private supabase: ReturnType<typeof createSupabaseClient>;

  constructor() {
    this.supabase = createSupabaseClient();
  }

  /**
   * Perform a similarity search using pgvector
   * @param request
   */
  async searchSimilarity(request: SimilaritySearchRequest): Promise<SimilaritySearchResult[]> {
    const preprocessed = textPreprocessor.preprocessText(request.query);

    if (preprocessed.isEmpty || !preprocessed.processed) {
      throw new Error('Query is empty or invalid after preprocessing');
    }

    // Generate embedding for query
    const queryEmbedding = await geminiEmbeddingService.generateEmbedding({
      text: preprocessed.processed,
      cache_result: true
    });

    // Convert embedding to PostgreSQL array format
    const embeddingArray = `[${queryEmbedding.embedding.join(',')}]`;

    // Use pgvector similarity search with cosine distance
    const { data, error } = await this.supabase
      .rpc('similarity_search', {
        query_embedding: embeddingArray,
        similarity_threshold: request.similarity_threshold || 0.7,
        match_count: request.limit || 20
      });

    if (error) {
      console.error('Error in similarity search:', error);
      throw error;
    }

    return data.map((record: any): SimilaritySearchResult => {
      return {
        chunk: {
          id: record.chunk_id,
          document_id: record.document_id,
          chunk_index: record.chunk_index,
          content: record.content,
          metadata: record.chunk_metadata || {},
          created_at: record.chunk_created_at
        },
        document: {
          id: record.document_id,
          user_id: record.user_id,
          title: record.document_title,
          file_name: record.file_name,
          file_type: record.file_type,
          content_preview: record.content_preview || '',
          processing_status: 'completed' as const,
          upload_path: record.upload_path || '',
          created_at: record.document_created_at,
          updated_at: record.document_updated_at
        },
        similarity_score: record.similarity,
        distance: 1 - record.similarity,
        metadata: {
          chunk_index: record.chunk_index,
          model_version: record.model_version,
          processing_time_ms: record.processing_time_ms
        },
        highlights: this.generateHighlights(record.content, request.query)
      };
    });
  }

  /**
   * Calculate cosine similarity
   * @param vectorA
   * @param vectorB
   */
  calculateCosineSimilarity(vectorA: number[], vectorB: number[]): number {
    const dotProduct = vectorA.reduce((sum, value, i) => sum + value * vectorB[i], 0);
    const magnitudeA = Math.sqrt(vectorA.reduce((sum, value) => sum + value * value, 0));
    const magnitudeB = Math.sqrt(vectorB.reduce((sum, value) => sum + value * value, 0));

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Generate text highlights for search results
   * @param content
   * @param query
   */
  generateHighlights(content: string, query: string): string[] {
    const highlights: string[] = [];
    const words = query.toLowerCase().split(/\s+/);
    const sentences = content.split(/[.!?]+/);

    for (const sentence of sentences) {
      const lowerSentence = sentence.toLowerCase();
      const matchCount = words.reduce((count, word) => {
        return count + (lowerSentence.includes(word) ? 1 : 0);
      }, 0);

      if (matchCount > 0) {
        highlights.push(sentence.trim());
      }
    }

    return highlights.slice(0, 3); // Return top 3 highlights
  }

  /**
   * Rank search results by relevance
   * @param results
   */
  rankSearchResults(results: SimilaritySearchResult[]): SimilaritySearchResult[] {
    return results.sort((a, b) => {
      // Primary sort by similarity score
      if (a.similarity_score !== b.similarity_score) {
        return b.similarity_score - a.similarity_score;
      }
      
      // Secondary sort by document creation date (newer first)
      return new Date(b.document.created_at).getTime() - new Date(a.document.created_at).getTime();
    });
  }

  /**
   * Cache search results
   * @param query
   * @param results
   */
  async cacheSearchResults(query: string, results: SimilaritySearchResult[]): Promise<void> {
    try {
      const crypto = await import('crypto');
      const queryHash = crypto.createHash('sha256').update(query).digest('hex');
      
      const expiresAt = new Date();
      expiresAt.setHours(expiresAt.getHours() + 1); // Cache for 1 hour

      await this.supabase
        .from('vector_search_cache')
        .upsert({
          query_hash: queryHash,
          query_text: query,
          search_results: results,
          similarity_threshold: 0.7,
          result_count: results.length,
          expires_at: expiresAt.toISOString()
        }, {
          onConflict: 'query_hash'
        });

    } catch (error) {
      console.error('Error caching search results:', error);
      // Don't throw - caching failure shouldn't break search
    }
  }

  /**
   * Get cached search results
   * @param query
   */
  async getCachedSearchResults(query: string): Promise<SimilaritySearchResult[] | null> {
    try {
      const crypto = await import('crypto');
      const queryHash = crypto.createHash('sha256').update(query).digest('hex');

      const { data, error } = await this.supabase
        .from('vector_search_cache')
        .select('search_results, expires_at')
        .eq('query_hash', queryHash)
        .gt('expires_at', new Date().toISOString())
        .single();

      if (error || !data) {
        return null;
      }

      // Update hit count
      await this.supabase
        .from('vector_search_cache')
        .update({
          hit_count: this.supabase.rpc('increment_hit_count'),
          last_accessed: new Date().toISOString()
        })
        .eq('query_hash', queryHash);

      return data.search_results as SimilaritySearchResult[];

    } catch (error) {
      console.error('Error getting cached search results:', error);
      return null;
    }
  }

  /**
   * Perform hybrid search (semantic + keyword)
   * @param request
   */
  async performHybridSearch(request: SimilaritySearchRequest & { keyword_weight?: number }): Promise<SimilaritySearchResult[]> {
    const keywordWeight = request.keyword_weight || 0.3;
    const semanticWeight = 1 - keywordWeight;

    // Get semantic search results
    const semanticResults = await this.searchSimilarity(request);

    // Get keyword search results
    const keywordResults = await this.keywordSearch(request.query, request.limit || 20);

    // Combine and rerank results
    const combinedResults = this.combineSearchResults(semanticResults, keywordResults, semanticWeight, keywordWeight);

    return this.rankSearchResults(combinedResults);
  }

  /**
   * Perform keyword search
   * @param query
   * @param limit
   */
  private async keywordSearch(query: string, limit: number): Promise<SimilaritySearchResult[]> {
    const { data, error } = await this.supabase
      .from('document_chunks')
      .select(`
        id,
        content,
        chunk_index,
        metadata,
        document_id,
        created_at,
        documents!inner (
          id,
          user_id,
          title,
          file_name,
          file_type,
          content_preview,
          upload_path,
          created_at,
          updated_at
        )
      `)
      .textSearch('content', query)
      .limit(limit);

    if (error) {
      console.error('Error in keyword search:', error);
      return [];
    }

    return data.map((record: any): SimilaritySearchResult => {
      return {
        chunk: {
          id: record.id,
          document_id: record.document_id,
          chunk_index: record.chunk_index,
          content: record.content,
          metadata: record.metadata || {},
          created_at: record.created_at
        },
        document: {
          id: record.documents.id,
          user_id: record.documents.user_id,
          title: record.documents.title,
          file_name: record.documents.file_name,
          file_type: record.documents.file_type,
          content_preview: record.documents.content_preview || '',
          processing_status: 'completed' as const,
          upload_path: record.documents.upload_path || '',
          created_at: record.documents.created_at,
          updated_at: record.documents.updated_at
        },
        similarity_score: 0.5, // Default score for keyword matches
        distance: 0.5,
        metadata: {
          chunk_index: record.chunk_index,
          model_version: 'keyword_search',
          processing_time_ms: 0
        },
        highlights: this.generateHighlights(record.content, query)
      };
    });
  }

  /**
   * Combine semantic and keyword search results
   * @param semanticResults
   * @param keywordResults
   * @param semanticWeight
   * @param keywordWeight
   */
  private combineSearchResults(
    semanticResults: SimilaritySearchResult[],
    keywordResults: SimilaritySearchResult[],
    semanticWeight: number,
    keywordWeight: number
  ): SimilaritySearchResult[] {
    const resultMap = new Map<string, SimilaritySearchResult>();

    // Add semantic results
    for (const result of semanticResults) {
      const key = `${result.chunk.document_id}-${result.chunk.chunk_index}`;
      resultMap.set(key, {
        ...result,
        similarity_score: result.similarity_score * semanticWeight
      });
    }

    // Add or combine keyword results
    for (const result of keywordResults) {
      const key = `${result.chunk.document_id}-${result.chunk.chunk_index}`;
      const existing = resultMap.get(key);
      
      if (existing) {
        // Combine scores
        existing.similarity_score += result.similarity_score * keywordWeight;
        existing.distance = 1 - existing.similarity_score;
      } else {
        resultMap.set(key, {
          ...result,
          similarity_score: result.similarity_score * keywordWeight
        });
      }
    }

    return Array.from(resultMap.values());
  }
}

// Export a singleton instance
export const similaritySearchService = new SimilaritySearchService();
export default similaritySearchService;
