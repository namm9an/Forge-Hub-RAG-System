import { supabaseAdmin } from '@/lib/supabase/server';
import { deleteFile } from '@/lib/storage/supabaseStorage';

export interface Document {
  id: string;
  user_id: string;
  title: string;
  file_name: string;
  file_type: string;
  file_size: number;
  content_preview?: string | null;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  upload_path?: string | null;
  chunk_count?: number;
  processing_error?: string | null;
  file_metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface CreateDocumentData {
  user_id: string;
  title: string;
  file_name: string;
  file_type: string;
  file_size: number;
  upload_path: string;
}

export interface DocumentFilter {
  userId?: string;
  status?: Document['processing_status'];
  dateFrom?: Date;
  dateTo?: Date;
  fileType?: string;
  searchQuery?: string;
}

export interface PaginationOptions {
  page?: number;
  limit?: number;
  orderBy?: 'created_at' | 'updated_at' | 'title' | 'file_size';
  orderDirection?: 'asc' | 'desc';
}

/**
 * Create a new document record
 */
export async function createDocument(data: CreateDocumentData): Promise<Document> {
  try {
    const { data: document, error } = await supabaseAdmin
      .from('documents')
      .insert({
        ...data,
        processing_status: 'pending',
        chunk_count: 0,
      })
      .select()
      .single();

    if (error) {
      throw new Error(`Failed to create document: ${error.message}`);
    }

    return document;
  } catch (error) {
    console.error('Error creating document:', error);
    throw error;
  }
}

/**
 * Update document status
 */
export async function updateDocumentStatus(
  id: string,
  status: Document['processing_status'],
  additionalData: Partial<Document> = {}
): Promise<void> {
  try {
    const { error } = await supabaseAdmin
      .from('documents')
      .update({
        processing_status: status,
        ...additionalData,
        updated_at: new Date().toISOString(),
      })
      .eq('id', id);

    if (error) {
      throw new Error(`Failed to update document status: ${error.message}`);
    }
  } catch (error) {
    console.error('Error updating document status:', error);
    throw error;
  }
}

/**
 * Get documents by user with filtering and pagination
 */
export async function getDocumentsByUser(
  userId: string,
  filter: DocumentFilter = {},
  pagination: PaginationOptions = {}
): Promise<{
  documents: Document[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}> {
  try {
    const {
      page = 1,
      limit = 20,
      orderBy = 'created_at',
      orderDirection = 'desc',
    } = pagination;

    // Build query
    let query = supabaseAdmin
      .from('documents')
      .select('*', { count: 'exact' })
      .eq('user_id', userId);

    // Apply filters
    if (filter.status) {
      query = query.eq('processing_status', filter.status);
    }

    if (filter.fileType) {
      query = query.eq('file_type', filter.fileType);
    }

    if (filter.dateFrom) {
      query = query.gte('created_at', filter.dateFrom.toISOString());
    }

    if (filter.dateTo) {
      query = query.lte('created_at', filter.dateTo.toISOString());
    }

    if (filter.searchQuery) {
      query = query.or(
        `title.ilike.%${filter.searchQuery}%,file_name.ilike.%${filter.searchQuery}%`
      );
    }

    // Apply pagination
    const offset = (page - 1) * limit;
    query = query
      .order(orderBy, { ascending: orderDirection === 'asc' })
      .range(offset, offset + limit - 1);

    const { data: documents, error, count } = await query;

    if (error) {
      throw new Error(`Failed to fetch documents: ${error.message}`);
    }

    const total = count || 0;
    const totalPages = Math.ceil(total / limit);

    return {
      documents: documents || [],
      total,
      page,
      pageSize: limit,
      totalPages,
    };
  } catch (error) {
    console.error('Error fetching documents:', error);
    throw error;
  }
}

/**
 * Get a single document by ID
 */
export async function getDocumentById(
  id: string,
  userId?: string
): Promise<Document | null> {
  try {
    let query = supabaseAdmin
      .from('documents')
      .select('*')
      .eq('id', id);

    // If userId is provided, ensure user owns the document
    if (userId) {
      query = query.eq('user_id', userId);
    }

    const { data: document, error } = await query.single();

    if (error) {
      if (error.code === 'PGRST116') {
        return null; // Document not found
      }
      throw new Error(`Failed to fetch document: ${error.message}`);
    }

    return document;
  } catch (error) {
    console.error('Error fetching document:', error);
    throw error;
  }
}

/**
 * Delete document and all related data
 */
export async function deleteDocumentAndChunks(
  id: string,
  userId?: string
): Promise<void> {
  try {
    // Get document first to verify ownership and get file path
    const document = await getDocumentById(id, userId);
    
    if (!document) {
      throw new Error('Document not found');
    }

    // Delete from storage if file exists
    if (document.upload_path) {
      try {
        await deleteFile(document.upload_path);
      } catch (error) {
        console.error('Error deleting file from storage:', error);
        // Continue with database deletion even if storage deletion fails
      }
    }

    // Delete document (cascades to chunks, embeddings, and processing jobs)
    const { error } = await supabaseAdmin
      .from('documents')
      .delete()
      .eq('id', id);

    if (error) {
      throw new Error(`Failed to delete document: ${error.message}`);
    }
  } catch (error) {
    console.error('Error deleting document:', error);
    throw error;
  }
}

/**
 * Get document statistics for a user
 */
export async function getUserDocumentStats(userId: string): Promise<{
  totalDocuments: number;
  totalSize: number;
  documentsByType: Record<string, number>;
  documentsByStatus: Record<string, number>;
  recentDocuments: Document[];
}> {
  try {
    // Get all documents for the user
    const { data: documents, error } = await supabaseAdmin
      .from('documents')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      throw new Error(`Failed to fetch document stats: ${error.message}`);
    }

    const docs = documents || [];

    // Calculate statistics
    const totalDocuments = docs.length;
    const totalSize = docs.reduce((sum, doc) => sum + doc.file_size, 0);

    // Group by file type
    const documentsByType = docs.reduce((acc, doc) => {
      const type = getFileTypeLabel(doc.file_type);
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Group by status
    const documentsByStatus = docs.reduce((acc, doc) => {
      acc[doc.processing_status] = (acc[doc.processing_status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Get recent documents (last 5)
    const recentDocuments = docs.slice(0, 5);

    return {
      totalDocuments,
      totalSize,
      documentsByType,
      documentsByStatus,
      recentDocuments,
    };
  } catch (error) {
    console.error('Error fetching document stats:', error);
    throw error;
  }
}

/**
 * Update document metadata
 */
export async function updateDocumentMetadata(
  id: string,
  metadata: Partial<Document>
): Promise<void> {
  try {
    const { error } = await supabaseAdmin
      .from('documents')
      .update({
        ...metadata,
        updated_at: new Date().toISOString(),
      })
      .eq('id', id);

    if (error) {
      throw new Error(`Failed to update document metadata: ${error.message}`);
    }
  } catch (error) {
    console.error('Error updating document metadata:', error);
    throw error;
  }
}

/**
 * Check if user has reached document limit
 */
export async function checkUserDocumentLimit(
  userId: string,
  limit: number = 100
): Promise<{
  withinLimit: boolean;
  currentCount: number;
  limit: number;
}> {
  try {
    const { count, error } = await supabaseAdmin
      .from('documents')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', userId);

    if (error) {
      throw new Error(`Failed to check document limit: ${error.message}`);
    }

    const currentCount = count || 0;

    return {
      withinLimit: currentCount < limit,
      currentCount,
      limit,
    };
  } catch (error) {
    console.error('Error checking document limit:', error);
    throw error;
  }
}

// Helper functions

function getFileTypeLabel(fileType: string): string {
  const typeMap: Record<string, string> = {
    'application/pdf': 'PDF',
    'text/plain': 'Text',
    'text/markdown': 'Markdown',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word',
  };

  return typeMap[fileType] || 'Other';
}
