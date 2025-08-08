import { supabaseAdmin } from '@/lib/supabase/server';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';

export interface UploadResult {
  path: string;
  url: string;
  size: number;
}

export class StorageError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly originalError?: Error
  ) {
    super(message);
    this.name = 'StorageError';
  }
}

/**
 * Upload a file to Supabase Storage
 * @param file - File to upload (as Buffer or Blob)
 * @param userId - User ID for organizing files
 * @param fileName - Original file name
 * @returns Upload result with path and URL
 */
export async function uploadFile(
  file: Buffer | Blob,
  userId: string,
  fileName: string
): Promise<UploadResult> {
  try {
    // Sanitize and prepare file path
    const sanitizedFileName = sanitizeFileName(fileName);
    const fileExtension = path.extname(sanitizedFileName);
    const uniqueFileName = `${uuidv4()}${fileExtension}`;
    const filePath = `${userId}/${uniqueFileName}`;

    // Get file size
    const fileSize = file instanceof Buffer ? file.length : file.size;

    // Check file size limit (50MB)
    const maxSize = 50 * 1024 * 1024;
    if (fileSize > maxSize) {
      throw new StorageError(
        `File size (${Math.round(fileSize / 1024 / 1024)}MB) exceeds maximum allowed size (50MB)`,
        'FILE_TOO_LARGE'
      );
    }

    // Upload to Supabase Storage
    const { data, error } = await supabaseAdmin.storage
      .from('documents')
      .upload(filePath, file, {
        contentType: getContentType(fileExtension),
        cacheControl: '3600',
        upsert: false,
      });

    if (error) {
      throw new StorageError(
        `Failed to upload file: ${error.message}`,
        'UPLOAD_ERROR',
        error
      );
    }

    // Get public URL (note: bucket is private, so this URL requires authentication)
    const url = getFileUrl(filePath);

    return {
      path: filePath,
      url,
      size: fileSize,
    };
  } catch (error) {
    if (error instanceof StorageError) {
      throw error;
    }
    throw new StorageError(
      `Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'UNKNOWN_ERROR',
      error as Error
    );
  }
}

/**
 * Download a file from Supabase Storage
 * @param filePath - Path to the file in storage
 * @returns File content as Buffer
 */
export async function downloadFile(filePath: string): Promise<Buffer> {
  try {
    const { data, error } = await supabaseAdmin.storage
      .from('documents')
      .download(filePath);

    if (error) {
      throw new StorageError(
        `Failed to download file: ${error.message}`,
        'DOWNLOAD_ERROR',
        error
      );
    }

    if (!data) {
      throw new StorageError(
        'No data received from storage',
        'EMPTY_RESPONSE'
      );
    }

    // Convert Blob to Buffer
    const buffer = Buffer.from(await data.arrayBuffer());
    return buffer;
  } catch (error) {
    if (error instanceof StorageError) {
      throw error;
    }
    throw new StorageError(
      `Download failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'UNKNOWN_ERROR',
      error as Error
    );
  }
}

/**
 * Delete a file from Supabase Storage
 * @param filePath - Path to the file in storage
 */
export async function deleteFile(filePath: string): Promise<void> {
  try {
    const { error } = await supabaseAdmin.storage
      .from('documents')
      .remove([filePath]);

    if (error) {
      throw new StorageError(
        `Failed to delete file: ${error.message}`,
        'DELETE_ERROR',
        error
      );
    }
  } catch (error) {
    if (error instanceof StorageError) {
      throw error;
    }
    throw new StorageError(
      `Delete failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'UNKNOWN_ERROR',
      error as Error
    );
  }
}

/**
 * Get the URL for a file in storage
 * @param filePath - Path to the file in storage
 * @returns Public URL (requires authentication to access)
 */
export function getFileUrl(filePath: string): string {
  const { data } = supabaseAdmin.storage
    .from('documents')
    .getPublicUrl(filePath);
  
  return data.publicUrl;
}

/**
 * Create a signed URL for temporary file access
 * @param filePath - Path to the file in storage
 * @param expiresIn - Expiration time in seconds (default: 1 hour)
 * @returns Signed URL for temporary access
 */
export async function createSignedUrl(
  filePath: string,
  expiresIn: number = 3600
): Promise<string> {
  try {
    const { data, error } = await supabaseAdmin.storage
      .from('documents')
      .createSignedUrl(filePath, expiresIn);

    if (error) {
      throw new StorageError(
        `Failed to create signed URL: ${error.message}`,
        'SIGNED_URL_ERROR',
        error
      );
    }

    if (!data?.signedUrl) {
      throw new StorageError(
        'No signed URL received',
        'EMPTY_RESPONSE'
      );
    }

    return data.signedUrl;
  } catch (error) {
    if (error instanceof StorageError) {
      throw error;
    }
    throw new StorageError(
      `Failed to create signed URL: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'UNKNOWN_ERROR',
      error as Error
    );
  }
}

/**
 * List files in a user's directory
 * @param userId - User ID
 * @param limit - Maximum number of files to return
 * @param offset - Offset for pagination
 * @returns List of files with metadata
 */
export async function listUserFiles(
  userId: string,
  limit: number = 100,
  offset: number = 0
): Promise<Array<{
  name: string;
  id?: string;
  created_at?: string;
  updated_at?: string;
  size?: number;
  metadata?: Record<string, any>;
}>> {
  try {
    const { data, error } = await supabaseAdmin.storage
      .from('documents')
      .list(userId, {
        limit,
        offset,
        sortBy: { column: 'created_at', order: 'desc' },
      });

    if (error) {
      throw new StorageError(
        `Failed to list files: ${error.message}`,
        'LIST_ERROR',
        error
      );
    }

    return data || [];
  } catch (error) {
    if (error instanceof StorageError) {
      throw error;
    }
    throw new StorageError(
      `Failed to list files: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'UNKNOWN_ERROR',
      error as Error
    );
  }
}

// Helper functions

/**
 * Sanitize file name for storage
 */
function sanitizeFileName(fileName: string): string {
  // Remove or replace problematic characters
  return fileName
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .replace(/_{2,}/g, '_')
    .replace(/^_|_$/g, '')
    .toLowerCase();
}

/**
 * Get content type based on file extension
 */
function getContentType(extension: string): string {
  const mimeTypes: Record<string, string> = {
    '.pdf': 'application/pdf',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  };

  return mimeTypes[extension.toLowerCase()] || 'application/octet-stream';
}

/**
 * Check if file exists in storage
 */
export async function fileExists(filePath: string): Promise<boolean> {
  try {
    const { data, error } = await supabaseAdmin.storage
      .from('documents')
      .list(path.dirname(filePath), {
        limit: 1,
        search: path.basename(filePath),
      });

    if (error) {
      return false;
    }

    return data && data.length > 0;
  } catch {
    return false;
  }
}

/**
 * Get file metadata from storage
 */
export async function getFileMetadata(filePath: string): Promise<{
  size: number;
  contentType: string;
  lastModified: string;
} | null> {
  try {
    const { data, error } = await supabaseAdmin.storage
      .from('documents')
      .list(path.dirname(filePath), {
        limit: 1,
        search: path.basename(filePath),
      });

    if (error || !data || data.length === 0) {
      return null;
    }

    const file = data[0];
    return {
      size: file.metadata?.size || 0,
      contentType: file.metadata?.mimetype || 'application/octet-stream',
      lastModified: file.updated_at || file.created_at || new Date().toISOString(),
    };
  } catch {
    return null;
  }
}
