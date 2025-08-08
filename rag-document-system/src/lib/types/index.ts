export interface User {
  id: string;
  email: string;
  full_name?: string;
  avatar_url?: string;
  created_at: string;
  updated_at: string;
}

export interface Profile {
  id: string;
  email: string;
  full_name?: string;
  avatar_url?: string;
  created_at: string;
  updated_at: string;
}

export interface Document {
  id: string;
  user_id: string;
  title: string;
  file_name: string;
  file_type: string;
  file_size: number;
  content_preview?: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  upload_path?: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  metadata: Record<string, any>;
  created_at: string;
}

export interface Embedding {
  id: string;
  chunk_id: string;
  embedding: number[];
  model_version: string;
  embedding_metadata: Record<string, any>;
  processing_time_ms?: number;
  similarity_cache: Record<string, any>;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error_message?: string;
  retry_count: number;
  created_at: string;
  updated_at: string;
}

export interface SearchHistory {
  id: string;
  user_id: string;
  query: string;
  results_count: number;
  created_at: string;
}

export interface AuthRequest {
  email: string;
  password: string;
}

export interface SignupRequest extends AuthRequest {
  full_name?: string;
}

export interface AuthResponse {
  user: User;
  token: string;
  expires_in: number;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T = any> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number;
}

export interface SystemInfo {
  version: string;
  environment: string;
  uptime: number;
  limits: {
    maxFileSize: number;
    maxDocuments: number;
    rateLimit: RateLimitInfo;
  };
}

export interface GeminiConfig {
  apiKey: string;
  model: string;
  embeddingModel: string;
  embeddingDimensions: number;
}

export interface ValidationError {
  field: string;
  message: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

export interface FileUploadResult {
  id: string;
  fileName: string;
  fileSize: number;
  fileType: string;
  uploadPath: string;
  status: 'uploaded' | 'processing' | 'failed';
}

export interface SearchQuery {
  query: string;
  limit?: number;
  offset?: number;
  filters?: Record<string, any>;
}

export interface SearchResult {
  document: Document;
  chunk: DocumentChunk;
  similarity: number;
  highlight?: string;
}

export interface HealthCheckResult {
  status: 'ok' | 'error';
  timestamp: string;
  services: {
    database: 'connected' | 'disconnected';
    gemini: 'connected' | 'disconnected';
    [key: string]: string;
  };
  uptime: number;
}

// Phase 3: Vector Embeddings Types
export interface EmbeddingJob {
  id: string;
  document_id: string;
  chunk_ids: string[];
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  batch_size: number;
  total_chunks: number;
  processed_chunks: number;
  failed_chunks: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  retry_count: number;
  estimated_completion?: string;
  processing_metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface EmbeddingJobProgress {
  id: string;
  document_id: string;
  document_title: string;
  file_name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  total_chunks: number;
  processed_chunks: number;
  failed_chunks: number;
  progress_percentage: number;
  started_at?: string;
  completed_at?: string;
  estimated_completion?: string;
  processing_time?: string;
  error_message?: string;
  retry_count: number;
}

export interface VectorSearchCache {
  id: string;
  user_id: string;
  query_hash: string;
  query_text: string;
  query_embedding?: number[];
  search_results: SearchResult[];
  similarity_threshold: number;
  result_count: number;
  search_metadata: Record<string, any>;
  hit_count: number;
  last_accessed: string;
  created_at: string;
  expires_at: string;
}

export interface EmbeddingCache {
  id: string;
  content_hash: string;
  content_preview: string;
  embedding: number[];
  model_version: string;
  usage_count: number;
  created_at: string;
  last_used: string;
}

export interface RateLimitTracking {
  id: string;
  service_name: string;
  user_id?: string;
  requests_count: number;
  tokens_used: number;
  reset_time: string;
  created_at: string;
  updated_at: string;
}

export interface VectorStats {
  total_embeddings: number;
  total_documents: number;
  avg_chunks_per_document: number;
  cache_hit_rate: number;
  storage_size_mb: number;
}

export interface SimilaritySearchRequest {
  query: string;
  similarity_threshold?: number;
  limit?: number;
  offset?: number;
  distance_metric?: 'cosine' | 'l2' | 'inner_product';
  include_metadata?: boolean;
  filters?: {
    document_ids?: string[];
    file_types?: string[];
    date_range?: {
      start: string;
      end: string;
    };
  };
}

export interface SimilaritySearchResult {
  chunk: DocumentChunk;
  document: Document;
  similarity_score: number;
  distance: number;
  metadata: {
    chunk_index: number;
    model_version: string;
    processing_time_ms?: number;
  };
  highlights?: string[];
}

export interface EmbeddingBatchRequest {
  document_ids: string[];
  batch_size?: number;
  priority?: 'low' | 'normal' | 'high';
  force_reprocess?: boolean;
}

export interface EmbeddingBatchResponse {
  job_id: string;
  total_documents: number;
  total_chunks: number;
  estimated_completion: string;
  batch_jobs: EmbeddingJob[];
}

export interface EmbeddingGenerationRequest {
  text: string;
  model_version?: string;
  cache_result?: boolean;
  metadata?: Record<string, any>;
}

export interface EmbeddingGenerationResponse {
  embedding: number[];
  model_version: string;
  processing_time_ms: number;
  dimensions: number;
  cached: boolean;
  metadata: Record<string, any>;
}

export interface VectorIndexStats {
  index_name: string;
  index_type: 'ivfflat' | 'hnsw';
  lists: number;
  probes: number;
  dimensions: number;
  total_vectors: number;
  index_size_mb: number;
  avg_query_time_ms: number;
  last_optimized: string;
  health_status: 'healthy' | 'degraded' | 'unhealthy';
}

export interface EmbeddingProcessingOptions {
  batch_size: number;
  max_retries: number;
  retry_delay_ms: number;
  rate_limit: {
    requests_per_minute: number;
    tokens_per_minute: number;
  };
  preprocessing: {
    clean_text: boolean;
    normalize_text: boolean;
    remove_duplicates: boolean;
    max_chunk_size: number;
  };
  caching: {
    enabled: boolean;
    ttl_hours: number;
    max_entries: number;
  };
}

export interface EmbeddingError {
  code: string;
  message: string;
  details?: Record<string, any>;
  retry_after?: number;
  chunk_id?: string;
  document_id?: string;
}

export interface EmbeddingMetrics {
  total_generated: number;
  total_cached: number;
  avg_processing_time_ms: number;
  cache_hit_rate: number;
  error_rate: number;
  rate_limit_hits: number;
  tokens_used: number;
  api_calls_made: number;
  storage_used_mb: number;
  period_start: string;
  period_end: string;
}
