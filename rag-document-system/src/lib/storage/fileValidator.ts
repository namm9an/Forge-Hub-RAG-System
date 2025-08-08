export interface ValidationResult {
  valid: boolean;
  error?: string;
}

export interface FileValidationOptions {
  maxSize?: number;
  allowedTypes?: string[];
  allowedExtensions?: string[];
}

const DEFAULT_OPTIONS: Required<FileValidationOptions> = {
  maxSize: 50 * 1024 * 1024, // 50MB
  allowedTypes: [
    'application/pdf',
    'text/plain',
    'text/markdown',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  ],
  allowedExtensions: ['.pdf', '.txt', '.md', '.docx'],
};

/**
 * Validate file type
 * @param file - File to validate
 * @param options - Validation options
 * @returns Validation result
 */
export function validateFileType(
  file: File | { type: string; name: string },
  options: FileValidationOptions = {}
): ValidationResult {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Check MIME type
  if (opts.allowedTypes.length > 0) {
    const fileType = file.type.toLowerCase();
    
    // Handle text/plain variations
    const normalizedType = normalizeFileType(fileType);
    
    if (!opts.allowedTypes.some(type => normalizedType === type.toLowerCase())) {
      // Check by extension if MIME type doesn't match
      const extension = getFileExtension(file.name).toLowerCase();
      
      if (!opts.allowedExtensions.includes(extension)) {
        return {
          valid: false,
          error: `File type "${file.type || 'unknown'}" is not supported. Allowed types: PDF, TXT, MD, DOCX`,
        };
      }
    }
  }

  // Check file extension
  if (opts.allowedExtensions.length > 0) {
    const extension = getFileExtension(file.name).toLowerCase();
    
    if (!opts.allowedExtensions.includes(extension)) {
      return {
        valid: false,
        error: `File extension "${extension}" is not supported. Allowed extensions: ${opts.allowedExtensions.join(', ')}`,
      };
    }
  }

  return { valid: true };
}

/**
 * Validate file size
 * @param file - File to validate
 * @param options - Validation options
 * @returns Validation result
 */
export function validateFileSize(
  file: File | { size: number },
  options: FileValidationOptions = {}
): ValidationResult {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  if (file.size > opts.maxSize) {
    const fileSizeMB = Math.round(file.size / 1024 / 1024);
    const maxSizeMB = Math.round(opts.maxSize / 1024 / 1024);
    
    return {
      valid: false,
      error: `File size (${fileSizeMB}MB) exceeds maximum allowed size (${maxSizeMB}MB)`,
    };
  }

  return { valid: true };
}

/**
 * Sanitize file name for safe storage
 * @param fileName - Original file name
 * @returns Sanitized file name
 */
export function sanitizeFileName(fileName: string): string {
  // Get file name without extension
  const extension = getFileExtension(fileName);
  const nameWithoutExt = fileName.slice(0, -extension.length);

  // Sanitize the name
  const sanitized = nameWithoutExt
    .trim()
    .replace(/[^\w\s.-]/g, '_') // Replace special chars with underscore
    .replace(/\s+/g, '_') // Replace spaces with underscore
    .replace(/_{2,}/g, '_') // Replace multiple underscores with single
    .replace(/^_|_$/g, '') // Remove leading/trailing underscores
    .slice(0, 100); // Limit length

  // Return with original extension
  return sanitized + extension;
}

/**
 * Validate a file comprehensively
 * @param file - File to validate
 * @param options - Validation options
 * @returns Validation result with all errors
 */
export function validateFile(
  file: File,
  options: FileValidationOptions = {}
): ValidationResult {
  // Check file existence
  if (!file) {
    return {
      valid: false,
      error: 'No file provided',
    };
  }

  // Check file name
  if (!file.name || file.name.trim().length === 0) {
    return {
      valid: false,
      error: 'File name is required',
    };
  }

  // Validate file type
  const typeValidation = validateFileType(file, options);
  if (!typeValidation.valid) {
    return typeValidation;
  }

  // Validate file size
  const sizeValidation = validateFileSize(file, options);
  if (!sizeValidation.valid) {
    return sizeValidation;
  }

  // Additional security checks
  const securityCheck = performSecurityChecks(file);
  if (!securityCheck.valid) {
    return securityCheck;
  }

  return { valid: true };
}

/**
 * Perform security checks on file
 */
function performSecurityChecks(file: File): ValidationResult {
  const fileName = file.name.toLowerCase();

  // Check for potentially dangerous file names
  const dangerousPatterns = [
    /^\./, // Hidden files
    /\.(exe|bat|cmd|com|pif|scr|vbs|js)$/i, // Executable files
    /\.(zip|rar|7z|tar|gz)$/i, // Archive files (if not expected)
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(fileName)) {
      return {
        valid: false,
        error: 'File type is not allowed for security reasons',
      };
    }
  }

  // Check for path traversal attempts
  if (fileName.includes('..') || fileName.includes('/') || fileName.includes('\\')) {
    return {
      valid: false,
      error: 'Invalid file name',
    };
  }

  return { valid: true };
}

/**
 * Get file extension from file name
 */
function getFileExtension(fileName: string): string {
  const lastDot = fileName.lastIndexOf('.');
  if (lastDot === -1) return '';
  return fileName.slice(lastDot).toLowerCase();
}

/**
 * Normalize file type for comparison
 */
function normalizeFileType(fileType: string): string {
  // Map common variations to standard types
  const typeMap: Record<string, string> = {
    'text/x-markdown': 'text/markdown',
    'text/md': 'text/markdown',
    'application/x-pdf': 'application/pdf',
  };

  return typeMap[fileType] || fileType;
}

/**
 * Get file type from extension
 */
export function getFileTypeFromExtension(fileName: string): string {
  const extension = getFileExtension(fileName);
  
  const extensionToType: Record<string, string> = {
    '.pdf': 'application/pdf',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  };

  return extensionToType[extension] || 'application/octet-stream';
}

/**
 * Check if file is text-based
 */
export function isTextFile(file: File | { type: string; name: string }): boolean {
  const textTypes = ['text/plain', 'text/markdown'];
  const textExtensions = ['.txt', '.md'];
  
  const fileType = file.type.toLowerCase();
  const extension = getFileExtension(file.name).toLowerCase();
  
  return textTypes.includes(fileType) || textExtensions.includes(extension);
}

/**
 * Estimate processing time based on file size and type
 * @returns Estimated time in seconds
 */
export function estimateProcessingTime(file: File): number {
  const sizeMB = file.size / (1024 * 1024);
  const baseTime = 5; // Base time in seconds
  
  // Different multipliers for different file types
  let typeMultiplier = 1;
  const extension = getFileExtension(file.name).toLowerCase();
  
  switch (extension) {
    case '.pdf':
      typeMultiplier = 2; // PDFs take longer
      break;
    case '.docx':
      typeMultiplier = 1.5; // DOCX moderate
      break;
    case '.txt':
    case '.md':
      typeMultiplier = 0.5; // Text files are fast
      break;
  }
  
  // Estimate: base time + (size in MB * type multiplier)
  return Math.ceil(baseTime + (sizeMB * typeMultiplier));
}
