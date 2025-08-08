import { GeminiClient } from '../gemini/client';
import { createSupabaseClient } from '../supabase/client';
import { RateLimiter } from 'rate-limiter-flexible';
import crypto from 'crypto';
import {
  EmbeddingGenerationRequest,
  EmbeddingGenerationResponse,
  EmbeddingError,
  EmbeddingCache,
  RateLimitTracking,
  EmbeddingProcessingOptions
} from '../types';

export class GeminiEmbeddingService {
  private client: GeminiClient;
  private supabase: ReturnType<typeof createSupabaseClient>;
  private rateLimiter: RateLimiter;
  private options: EmbeddingProcessingOptions;

  constructor(options?: Partial<EmbeddingProcessingOptions>) {
    this.client = new GeminiClient();
    this.supabase = createSupabaseClient();
    
    this.options = {
      batch_size: 10,
      max_retries: 3,
      retry_delay_ms: 4000,
      rate_limit: {
        requests_per_minute: 15,
        tokens_per_minute: 32000
      },
      preprocessing: {
        clean_text: true,
        normalize_text: true,
        remove_duplicates: true,
        max_chunk_size: 8192
      },
      caching: {
        enabled: true,
        ttl_hours: 24,
        max_entries: 10000
      },
      ...options
    };

    // Initialize rate limiter
    this.rateLimiter = new RateLimiter({
      points: this.options.rate_limit.requests_per_minute,
      duration: 60, // 1 minute
      blockDuration: 60 // Block for 1 minute if limit exceeded
    });
  }

  /**
   * Generate embedding for a single text with comprehensive error handling
   */
  async generateEmbedding(request: EmbeddingGenerationRequest): Promise<EmbeddingGenerationResponse> {
    const startTime = Date.now();
    
    try {
      // Validate and preprocess text
      const processedText = this.preprocessText(request.text);
      if (!processedText) {
        throw new EmbeddingError({
          code: 'INVALID_TEXT',
          message: 'Text cannot be empty after preprocessing',
          details: { originalText: request.text }
        });
      }

      // Check cache first if enabled
      if (this.options.caching.enabled && request.cache_result !== false) {
        const cached = await this.getCachedEmbedding(processedText);
        if (cached) {
          return {
            embedding: cached.embedding,
            model_version: cached.model_version,
            processing_time_ms: Date.now() - startTime,
            dimensions: cached.embedding.length,
            cached: true,
            metadata: request.metadata || {}
          };
        }
      }

      // Check rate limit
      await this.checkRateLimit();

      // Generate embedding
      const embedding = await this.generateEmbeddingWithRetry(processedText);
      
      // Cache the result
      if (this.options.caching.enabled && request.cache_result !== false) {
        await this.cacheEmbedding(processedText, embedding, request.model_version);
      }

      // Track usage
      await this.trackUsage(1, this.estimateTokens(processedText));

      return {
        embedding,
        model_version: request.model_version || 'text-embedding-004',
        processing_time_ms: Date.now() - startTime,
        dimensions: embedding.length,
        cached: false,
        metadata: request.metadata || {}
      };

    } catch (error) {
      const processingTime = Date.now() - startTime;
      console.error('Embedding generation failed:', error);
      
      if (error instanceof EmbeddingError) {
        throw error;
      }
      
      throw new EmbeddingError({
        code: 'GENERATION_FAILED',
        message: `Failed to generate embedding: ${error instanceof Error ? error.message : 'Unknown error'}`,
        details: { 
          processingTime,
          text: request.text.substring(0, 100) + '...',
          error: error instanceof Error ? error.message : String(error)
        }
      });
    }
  }

  /**
   * Generate embeddings for multiple texts in batches
   */
  async generateBatchEmbeddings(texts: string[]): Promise<EmbeddingGenerationResponse[]> {
    const results: EmbeddingGenerationResponse[] = [];
    const batches = this.createBatches(texts, this.options.batch_size);

    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      console.log(`Processing batch ${i + 1}/${batches.length} (${batch.length} items)`);

      const batchPromises = batch.map(text => 
        this.generateEmbedding({ text, cache_result: true })
      );

      try {
        const batchResults = await Promise.allSettled(batchPromises);
        
        for (const result of batchResults) {
          if (result.status === 'fulfilled') {
            results.push(result.value);
          } else {
            console.error('Batch item failed:', result.reason);
            // Add error placeholder
            results.push({
              embedding: [],
              model_version: 'text-embedding-004',
              processing_time_ms: 0,
              dimensions: 0,
              cached: false,
              metadata: { error: result.reason.message }
            });
          }
        }

        // Delay between batches to respect rate limits
        if (i < batches.length - 1) {
          await this.delay(this.options.retry_delay_ms);
        }

      } catch (error) {
        console.error(`Batch ${i + 1} failed:`, error);
        // Add error placeholders for entire batch
        for (let j = 0; j < batch.length; j++) {
          results.push({
            embedding: [],
            model_version: 'text-embedding-004',
            processing_time_ms: 0,
            dimensions: 0,
            cached: false,
            metadata: { error: error instanceof Error ? error.message : 'Batch failed' }
          });
        }
      }
    }

    return results;
  }

  /**
   * Generate embedding with retry logic
   */
  private async generateEmbeddingWithRetry(text: string): Promise<number[]> {
    let lastError: Error | null = null;
    
    for (let attempt = 1; attempt <= this.options.max_retries; attempt++) {
      try {
        const embedding = await this.client.generateEmbedding(text);
        
        // Validate embedding
        if (!embedding || embedding.length === 0) {
          throw new Error('Empty embedding received');
        }
        
        if (embedding.length !== 768) {
          throw new Error(`Invalid embedding dimensions: ${embedding.length}, expected 768`);
        }

        return embedding;

      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        if (this.isRateLimitError(error)) {
          const retryAfter = this.extractRetryAfter(error) || this.options.retry_delay_ms;
          console.warn(`Rate limit hit, retrying after ${retryAfter}ms (attempt ${attempt})`);
          await this.delay(retryAfter);
          continue;
        }
        
        if (attempt === this.options.max_retries) {
          break;
        }
        
        // Exponential backoff
        const backoffDelay = this.options.retry_delay_ms * Math.pow(2, attempt - 1);
        console.warn(`Attempt ${attempt} failed, retrying after ${backoffDelay}ms:`, error);
        await this.delay(backoffDelay);
      }
    }

    throw lastError || new Error('Max retries exceeded');
  }

  /**
   * Preprocess text for embedding generation
   */
  private preprocessText(text: string): string {
    if (!text || typeof text !== 'string') {
      return '';
    }

    let processed = text;

    if (this.options.preprocessing.clean_text) {
      // Remove excessive whitespace
      processed = processed.replace(/\s+/g, ' ').trim();
      
      // Remove special characters that might cause issues
      processed = processed.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g, '');
    }

    if (this.options.preprocessing.normalize_text) {
      // Normalize unicode characters
      processed = processed.normalize('NFKC');
    }

    // Truncate to max chunk size
    if (processed.length > this.options.preprocessing.max_chunk_size) {
      processed = processed.substring(0, this.options.preprocessing.max_chunk_size);
      // Try to break at word boundary
      const lastSpace = processed.lastIndexOf(' ');
      if (lastSpace > processed.length * 0.8) {
        processed = processed.substring(0, lastSpace);
      }
    }

    return processed;
  }

  /**
   * Check and get cached embedding
   */
  private async getCachedEmbedding(text: string): Promise<EmbeddingCache | null> {
    try {
      const contentHash = this.generateContentHash(text);
      
      const { data, error } = await this.supabase
        .from('embedding_cache')
        .select('*')
        .eq('content_hash', contentHash)
        .single();

      if (error || !data) {
        return null;
      }

      // Update usage count and last_used
      await this.supabase
        .from('embedding_cache')
        .update({
          usage_count: data.usage_count + 1,
          last_used: new Date().toISOString()
        })
        .eq('id', data.id);

      return data;

    } catch (error) {
      console.error('Error checking cache:', error);
      return null;
    }
  }

  /**
   * Cache embedding result
   */
  private async cacheEmbedding(text: string, embedding: number[], modelVersion = 'text-embedding-004'): Promise<void> {
    try {
      const contentHash = this.generateContentHash(text);
      const contentPreview = text.substring(0, 200);

      await this.supabase
        .from('embedding_cache')
        .upsert({
          content_hash: contentHash,
          content_preview: contentPreview,
          embedding,
          model_version: modelVersion,
          usage_count: 1,
          last_used: new Date().toISOString()
        }, {
          onConflict: 'content_hash'
        });

    } catch (error) {
      console.error('Error caching embedding:', error);
      // Don't throw - caching failure shouldn't break the flow
    }
  }

  /**
   * Check rate limit and wait if necessary
   */
  private async checkRateLimit(): Promise<void> {
    try {
      await this.rateLimiter.consume('embedding_generation');
    } catch (rejRes) {
      // Rate limit exceeded, wait
      const waitTime = Math.round(rejRes.msBeforeNext) || this.options.retry_delay_ms;
      console.warn(`Rate limit exceeded, waiting ${waitTime}ms`);
      await this.delay(waitTime);
    }
  }

  /**
   * Track API usage for monitoring
   */
  private async trackUsage(requestCount: number, tokensUsed: number): Promise<void> {
    try {
      const resetTime = new Date();
      resetTime.setMinutes(resetTime.getMinutes() + 1); // Reset every minute

      await this.supabase
        .from('rate_limit_tracking')
        .upsert({
          service_name: 'gemini_embeddings',
          requests_count: requestCount,
          tokens_used: tokensUsed,
          reset_time: resetTime.toISOString()
        }, {
          onConflict: 'service_name,user_id,reset_time'
        });

    } catch (error) {
      console.error('Error tracking usage:', error);
    }
  }

  /**
   * Utility functions
   */
  private generateContentHash(text: string): string {
    return crypto.createHash('sha256').update(text).digest('hex');
  }

  private createBatches<T>(array: T[], batchSize: number): T[][] {
    const batches: T[][] = [];
    for (let i = 0; i < array.length; i += batchSize) {
      batches.push(array.slice(i, i + batchSize));
    }
    return batches;
  }

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private isRateLimitError(error: any): boolean {
    const errorMessage = error?.message?.toLowerCase() || '';
    return errorMessage.includes('rate limit') || 
           errorMessage.includes('quota') || 
           errorMessage.includes('429') ||
           error?.status === 429;
  }

  private extractRetryAfter(error: any): number | null {
    if (error?.headers?.['retry-after']) {
      return parseInt(error.headers['retry-after']) * 1000;
    }
    return null;
  }

  private estimateTokens(text: string): number {
    // Rough estimation: 1 token ≈ 4 characters
    return Math.ceil(text.length / 4);
  }

  /**
   * Validate embedding dimensions and values
   */
  validateEmbedding(embedding: number[]): boolean {
    if (!Array.isArray(embedding) || embedding.length !== 768) {
      return false;
    }
    
    return embedding.every(value => 
      typeof value === 'number' && 
      !isNaN(value) && 
      isFinite(value)
    );
  }

  /**
   * Get service configuration
   */
  getConfiguration(): EmbeddingProcessingOptions {
    return { ...this.options };
  }

  /**
   * Update service configuration
   */
  updateConfiguration(newOptions: Partial<EmbeddingProcessingOptions>): void {
    this.options = { ...this.options, ...newOptions };
  }

  /**
   * Test service connectivity
   */
  async testConnection(): Promise<boolean> {
    try {
      const testResponse = await this.generateEmbedding({
        text: 'Test connection',
        cache_result: false
      });
      
      return testResponse.embedding.length === 768;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  }
}

// Export singleton instance
export const geminiEmbeddingService = new GeminiEmbeddingService();
export default geminiEmbeddingService;
