import { supabaseAdmin } from '@/lib/supabase/server';
import type { DocumentChunk, ChunkMetadata } from '@/lib/processing/textChunker';

export interface DatabaseChunk {
  id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  metadata: ChunkMetadata;
  created_at: string;
}

export interface ChunkFilter {
  documentId?: string;
  hasCode?: boolean;
  chunkType?: ChunkMetadata['chunk_type'];
  searchQuery?: string;
}

/**
 * Create document chunks in batch
 */
export async function createDocumentChunks(
  chunks: Array<{
    document_id: string;
    chunk_index: number;
    content: string;
    metadata: ChunkMetadata;
  }>
): Promise<void> {
  try {
    // Insert in batches to avoid hitting database limits
    const batchSize = 100;
    
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      
      const { error } = await supabaseAdmin
        .from('document_chunks')
        .insert(batch);

      if (error) {
        throw new Error(`Failed to insert chunks batch ${i / batchSize + 1}: ${error.message}`);
      }
    }
  } catch (error) {
    console.error('Error creating document chunks:', error);
    throw error;
  }
}

/**
 * Get chunks by document ID
 */
export async function getChunksByDocument(
  documentId: string,
  limit?: number,
  offset?: number
): Promise<DatabaseChunk[]> {
  try {
    let query = supabaseAdmin
      .from('document_chunks')
      .select('*')
      .eq('document_id', documentId)
      .order('chunk_index', { ascending: true });

    if (limit !== undefined) {
      query = query.limit(limit);
    }

    if (offset !== undefined) {
      query = query.range(offset, offset + (limit || 50) - 1);
    }

    const { data: chunks, error } = await query;

    if (error) {
      throw new Error(`Failed to fetch chunks: ${error.message}`);
    }

    return chunks || [];
  } catch (error) {
    console.error('Error fetching chunks:', error);
    throw error;
  }
}

/**
 * Get a specific chunk by ID
 */
export async function getChunkById(chunkId: string): Promise<DatabaseChunk | null> {
  try {
    const { data: chunk, error } = await supabaseAdmin
      .from('document_chunks')
      .select('*')
      .eq('id', chunkId)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return null; // Chunk not found
      }
      throw new Error(`Failed to fetch chunk: ${error.message}`);
    }

    return chunk;
  } catch (error) {
    console.error('Error fetching chunk:', error);
    throw error;
  }
}

/**
 * Update chunk metadata
 */
export async function updateChunkMetadata(
  chunkId: string,
  metadata: Partial<ChunkMetadata>
): Promise<void> {
  try {
    // Get existing chunk
    const { data: chunk, error: fetchError } = await supabaseAdmin
      .from('document_chunks')
      .select('metadata')
      .eq('id', chunkId)
      .single();

    if (fetchError) {
      throw new Error(`Failed to fetch chunk: ${fetchError.message}`);
    }

    // Merge metadata
    const updatedMetadata = {
      ...chunk.metadata,
      ...metadata,
    };

    // Update chunk
    const { error: updateError } = await supabaseAdmin
      .from('document_chunks')
      .update({ metadata: updatedMetadata })
      .eq('id', chunkId);

    if (updateError) {
      throw new Error(`Failed to update chunk metadata: ${updateError.message}`);
    }
  } catch (error) {
    console.error('Error updating chunk metadata:', error);
    throw error;
  }
}

/**
 * Search chunks with filters
 */
export async function searchChunks(
  userId: string,
  filter: ChunkFilter,
  limit: number = 50,
  offset: number = 0
): Promise<{
  chunks: Array<DatabaseChunk & { document_title?: string }>;
  total: number;
}> {
  try {
    // Build complex query with join
    let query = supabaseAdmin
      .from('document_chunks')
      .select(`
        *,
        documents!inner(
          id,
          title,
          user_id
        )
      `, { count: 'exact' })
      .eq('documents.user_id', userId);

    // Apply filters
    if (filter.documentId) {
      query = query.eq('document_id', filter.documentId);
    }

    if (filter.hasCode !== undefined) {
      query = query.eq('metadata->>has_code', filter.hasCode.toString());
    }

    if (filter.chunkType) {
      query = query.eq('metadata->>chunk_type', filter.chunkType);
    }

    if (filter.searchQuery) {
      query = query.ilike('content', `%${filter.searchQuery}%`);
    }

    // Apply pagination
    query = query
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    const { data, error, count } = await query;

    if (error) {
      throw new Error(`Failed to search chunks: ${error.message}`);
    }

    // Transform results to include document title
    const chunks = (data || []).map((item: any) => ({
      id: item.id,
      document_id: item.document_id,
      chunk_index: item.chunk_index,
      content: item.content,
      metadata: item.metadata,
      created_at: item.created_at,
      document_title: item.documents?.title,
    }));

    return {
      chunks,
      total: count || 0,
    };
  } catch (error) {
    console.error('Error searching chunks:', error);
    throw error;
  }
}

/**
 * Get chunk count for a document
 */
export async function getDocumentChunkCount(documentId: string): Promise<number> {
  try {
    const { count, error } = await supabaseAdmin
      .from('document_chunks')
      .select('*', { count: 'exact', head: true })
      .eq('document_id', documentId);

    if (error) {
      throw new Error(`Failed to count chunks: ${error.message}`);
    }

    return count || 0;
  } catch (error) {
    console.error('Error counting chunks:', error);
    throw error;
  }
}

/**
 * Delete all chunks for a document
 */
export async function deleteDocumentChunks(documentId: string): Promise<number> {
  try {
    const { data, error } = await supabaseAdmin
      .from('document_chunks')
      .delete()
      .eq('document_id', documentId)
      .select();

    if (error) {
      throw new Error(`Failed to delete chunks: ${error.message}`);
    }

    return data?.length || 0;
  } catch (error) {
    console.error('Error deleting chunks:', error);
    throw error;
  }
}

/**
 * Get chunks with embeddings for a document
 */
export async function getChunksWithEmbeddings(
  documentId: string
): Promise<Array<DatabaseChunk & { embedding_id?: string }>> {
  try {
    const { data, error } = await supabaseAdmin
      .from('document_chunks')
      .select(`
        *,
        embeddings(id)
      `)
      .eq('document_id', documentId)
      .order('chunk_index', { ascending: true });

    if (error) {
      throw new Error(`Failed to fetch chunks with embeddings: ${error.message}`);
    }

    return (data || []).map((item: any) => ({
      ...item,
      embedding_id: item.embeddings?.[0]?.id,
    }));
  } catch (error) {
    console.error('Error fetching chunks with embeddings:', error);
    throw error;
  }
}

/**
 * Get chunk statistics for a user
 */
export async function getUserChunkStats(userId: string): Promise<{
  totalChunks: number;
  chunksByType: Record<string, number>;
  averageChunkSize: number;
  documentsWithChunks: number;
}> {
  try {
    // Get all chunks for user's documents
    const { data: chunks, error } = await supabaseAdmin
      .from('document_chunks')
      .select(`
        *,
        documents!inner(user_id)
      `)
      .eq('documents.user_id', userId);

    if (error) {
      throw new Error(`Failed to fetch chunk stats: ${error.message}`);
    }

    const chunkData = chunks || [];
    
    // Calculate statistics
    const totalChunks = chunkData.length;
    
    // Group by chunk type
    const chunksByType = chunkData.reduce((acc, chunk) => {
      const type = chunk.metadata?.chunk_type || 'unknown';
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Calculate average chunk size
    const totalSize = chunkData.reduce((sum, chunk) => 
      sum + (chunk.metadata?.char_count || 0), 0
    );
    const averageChunkSize = totalChunks > 0 ? Math.round(totalSize / totalChunks) : 0;

    // Count unique documents
    const uniqueDocuments = new Set(chunkData.map(chunk => chunk.document_id));
    const documentsWithChunks = uniqueDocuments.size;

    return {
      totalChunks,
      chunksByType,
      averageChunkSize,
      documentsWithChunks,
    };
  } catch (error) {
    console.error('Error fetching chunk stats:', error);
    throw error;
  }
}
