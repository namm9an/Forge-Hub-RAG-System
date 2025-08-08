import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';
import fs from 'fs/promises';
import path from 'path';

export interface ParsedDocument {
  text: string;
  metadata: DocumentMetadata;
}

export interface DocumentMetadata {
  pageCount?: number;
  title?: string;
  author?: string;
  creationDate?: Date;
  fileSize: number;
  encoding?: string;
}

export class DocumentParserError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly originalError?: Error
  ) {
    super(message);
    this.name = 'DocumentParserError';
  }
}

/**
 * Parse document from file path with comprehensive error handling
 * @param filePath - Path to the document file
 * @param fileType - Type of the document (pdf, docx, txt, md)
 * @returns Parsed document with text content and metadata
 */
export async function parseDocument(
  filePath: string,
  fileType: string
): Promise<ParsedDocument> {
  try {
    // Validate file exists
    try {
      await fs.access(filePath);
    } catch (error) {
      throw new DocumentParserError(
        `File not found: ${filePath}`,
        'FILE_NOT_FOUND',
        error as Error
      );
    }

    // Get file stats
    const stats = await fs.stat(filePath);
    const fileSize = stats.size;

    // Check file size limit (50MB)
    const maxSize = 50 * 1024 * 1024;
    if (fileSize > maxSize) {
      throw new DocumentParserError(
        `File size (${Math.round(fileSize / 1024 / 1024)}MB) exceeds maximum allowed size (50MB)`,
        'FILE_TOO_LARGE'
      );
    }

    // Read file into buffer
    const buffer = await fs.readFile(filePath);

    // Parse based on file type
    const normalizedType = fileType.toLowerCase().replace('.', '');
    let result: ParsedDocument;

    switch (normalizedType) {
      case 'pdf':
      case 'application/pdf':
        result = await parsePDF(buffer, fileSize);
        break;
      case 'docx':
      case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        result = await parseDocx(buffer, fileSize);
        break;
      case 'txt':
      case 'text/plain':
      case 'md':
      case 'text/markdown':
        result = await parseText(buffer, fileSize);
        break;
      default:
        throw new DocumentParserError(
          `Unsupported file type: ${fileType}`,
          'UNSUPPORTED_FILE_TYPE'
        );
    }

    // Validate parsed content
    if (!result.text || result.text.trim().length === 0) {
      throw new DocumentParserError(
        'Document appears to be empty or could not extract text',
        'EMPTY_DOCUMENT'
      );
    }

    return result;
  } catch (error) {
    if (error instanceof DocumentParserError) {
      throw error;
    }
    throw new DocumentParserError(
      `Failed to parse document: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'PARSE_ERROR',
      error as Error
    );
  }
}

/**
 * Parse PDF document with metadata extraction
 */
export async function parsePDF(
  buffer: Buffer,
  fileSize: number
): Promise<ParsedDocument> {
  try {
    const data = await pdfParse(buffer, {
      max: 0, // No page limit
      version: 'v1.10.100'
    });

    const metadata: DocumentMetadata = {
      fileSize,
      pageCount: data.numpages,
      title: data.info?.Title,
      author: data.info?.Author,
      creationDate: data.info?.CreationDate
        ? new Date(data.info.CreationDate)
        : undefined,
    };

    return {
      text: data.text,
      metadata,
    };
  } catch (error) {
    throw new DocumentParserError(
      `PDF parsing failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'PDF_PARSE_ERROR',
      error as Error
    );
  }
}

/**
 * Parse DOCX document with mammoth
 */
export async function parseDocx(
  buffer: Buffer,
  fileSize: number
): Promise<ParsedDocument> {
  try {
    const result = await mammoth.extractRawText({ buffer });

    if (result.messages && result.messages.length > 0) {
      console.warn('DOCX parsing warnings:', result.messages);
    }

    const metadata: DocumentMetadata = {
      fileSize,
      // DOCX metadata extraction would require additional libraries
      // For now, we'll just include basic info
    };

    return {
      text: result.value,
      metadata,
    };
  } catch (error) {
    throw new DocumentParserError(
      `DOCX parsing failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'DOCX_PARSE_ERROR',
      error as Error
    );
  }
}

/**
 * Parse plain text or markdown files
 */
export async function parseText(
  buffer: Buffer,
  fileSize: number
): Promise<ParsedDocument> {
  try {
    // Detect encoding (simple detection, could be enhanced)
    const text = buffer.toString('utf-8');

    // Basic validation for text encoding
    if (text.includes('�')) {
      console.warn('Possible encoding issue detected in text file');
    }

    const metadata: DocumentMetadata = {
      fileSize,
      encoding: 'utf-8',
    };

    return {
      text,
      metadata,
    };
  } catch (error) {
    throw new DocumentParserError(
      `Text parsing failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'TEXT_PARSE_ERROR',
      error as Error
    );
  }
}

/**
 * Parse document from buffer (used for direct uploads)
 */
export async function parseDocumentFromBuffer(
  buffer: Buffer,
  fileType: string,
  fileName: string
): Promise<ParsedDocument> {
  const fileSize = buffer.length;

  // Check file size limit (50MB)
  const maxSize = 50 * 1024 * 1024;
  if (fileSize > maxSize) {
    throw new DocumentParserError(
      `File size (${Math.round(fileSize / 1024 / 1024)}MB) exceeds maximum allowed size (50MB)`,
      'FILE_TOO_LARGE'
    );
  }

  const normalizedType = fileType.toLowerCase();
  let result: ParsedDocument;

  switch (normalizedType) {
    case 'application/pdf':
      result = await parsePDF(buffer, fileSize);
      break;
    case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
      result = await parseDocx(buffer, fileSize);
      break;
    case 'text/plain':
    case 'text/markdown':
      result = await parseText(buffer, fileSize);
      break;
    default:
      // Try to infer from file extension
      const ext = path.extname(fileName).toLowerCase();
      switch (ext) {
        case '.pdf':
          result = await parsePDF(buffer, fileSize);
          break;
        case '.docx':
          result = await parseDocx(buffer, fileSize);
          break;
        case '.txt':
        case '.md':
          result = await parseText(buffer, fileSize);
          break;
        default:
          throw new DocumentParserError(
            `Unsupported file type: ${fileType}`,
            'UNSUPPORTED_FILE_TYPE'
          );
      }
  }

  if (!result.text || result.text.trim().length === 0) {
    throw new DocumentParserError(
      'Document appears to be empty or could not extract text',
      'EMPTY_DOCUMENT'
    );
  }

  return result;
}

