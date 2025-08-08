import crypto from 'crypto';

export interface TextPreprocessingOptions {
  cleanText: boolean;
  normalizeText: boolean;
  removeExcessiveWhitespace: boolean;
  removeSpecialCharacters: boolean;
  normalizeUnicode: boolean;
  maxChunkSize: number;
  preserveLineBreaks: boolean;
  removeHtmlTags: boolean;
  removeUrls: boolean;
  removeEmails: boolean;
  removeNumbers: boolean;
  toLowerCase: boolean;
}

export interface PreprocessingResult {
  original: string;
  processed: string;
  modifications: string[];
  characterCount: number;
  wordCount: number;
  contentHash: string;
  truncated: boolean;
  isEmpty: boolean;
}

export class TextPreprocessor {
  private options: TextPreprocessingOptions;

  constructor(options: Partial<TextPreprocessingOptions> = {}) {
    this.options = {
      cleanText: true,
      normalizeText: true,
      removeExcessiveWhitespace: true,
      removeSpecialCharacters: false,
      normalizeUnicode: true,
      maxChunkSize: 8192,
      preserveLineBreaks: false,
      removeHtmlTags: true,
      removeUrls: false,
      removeEmails: false,
      removeNumbers: false,
      toLowerCase: false,
      ...options
    };
  }

  /**
   * Preprocess text for embedding generation
   */
  preprocessText(text: string): PreprocessingResult {
    if (!text || typeof text !== 'string') {
      return {
        original: text || '',
        processed: '',
        modifications: ['empty_input'],
        characterCount: 0,
        wordCount: 0,
        contentHash: this.generateHash(''),
        truncated: false,
        isEmpty: true
      };
    }

    let processed = text;
    const modifications: string[] = [];

    // Store original for comparison
    const original = text;

    // Remove HTML tags
    if (this.options.removeHtmlTags) {
      const beforeHtml = processed;
      processed = this.removeHtmlTags(processed);
      if (beforeHtml !== processed) {
        modifications.push('html_tags_removed');
      }
    }

    // Remove URLs
    if (this.options.removeUrls) {
      const beforeUrls = processed;
      processed = this.removeUrls(processed);
      if (beforeUrls !== processed) {
        modifications.push('urls_removed');
      }
    }

    // Remove email addresses
    if (this.options.removeEmails) {
      const beforeEmails = processed;
      processed = this.removeEmails(processed);
      if (beforeEmails !== processed) {
        modifications.push('emails_removed');
      }
    }

    // Remove or normalize numbers
    if (this.options.removeNumbers) {
      const beforeNumbers = processed;
      processed = this.removeNumbers(processed);
      if (beforeNumbers !== processed) {
        modifications.push('numbers_removed');
      }
    }

    // Normalize unicode characters
    if (this.options.normalizeUnicode) {
      const beforeUnicode = processed;
      processed = processed.normalize('NFKC');
      if (beforeUnicode !== processed) {
        modifications.push('unicode_normalized');
      }
    }

    // Remove excessive whitespace
    if (this.options.removeExcessiveWhitespace) {
      const beforeWhitespace = processed;
      if (this.options.preserveLineBreaks) {
        // Preserve line breaks but remove excessive spaces
        processed = processed.replace(/[^\S\r\n]+/g, ' ');
        processed = processed.replace(/\n\s*\n/g, '\n\n'); // Max 2 consecutive newlines
      } else {
        processed = processed.replace(/\s+/g, ' ');
      }
      processed = processed.trim();
      
      if (beforeWhitespace !== processed) {
        modifications.push('whitespace_normalized');
      }
    }

    // Remove special characters (control characters, etc.)
    if (this.options.removeSpecialCharacters) {
      const beforeSpecial = processed;
      processed = processed.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g, '');
      if (beforeSpecial !== processed) {
        modifications.push('special_characters_removed');
      }
    }

    // Convert to lowercase
    if (this.options.toLowerCase) {
      const beforeLower = processed;
      processed = processed.toLowerCase();
      if (beforeLower !== processed) {
        modifications.push('converted_to_lowercase');
      }
    }

    // Truncate to max chunk size
    let truncated = false;
    if (processed.length > this.options.maxChunkSize) {
      processed = this.truncateText(processed, this.options.maxChunkSize);
      truncated = true;
      modifications.push('truncated');
    }

    // Additional cleaning if enabled
    if (this.options.cleanText) {
      const beforeClean = processed;
      processed = this.cleanText(processed);
      if (beforeClean !== processed) {
        modifications.push('text_cleaned');
      }
    }

    // Final validation
    if (!processed || processed.trim().length === 0) {
      return {
        original,
        processed: '',
        modifications: [...modifications, 'empty_after_processing'],
        characterCount: 0,
        wordCount: 0,
        contentHash: this.generateHash(''),
        truncated,
        isEmpty: true
      };
    }

    return {
      original,
      processed,
      modifications,
      characterCount: processed.length,
      wordCount: this.countWords(processed),
      contentHash: this.generateHash(processed),
      truncated,
      isEmpty: false
    };
  }

  /**
   * Clean text by removing unwanted characters and patterns
   */
  private cleanText(text: string): string {
    let cleaned = text;

    // Remove excessive punctuation
    cleaned = cleaned.replace(/[.]{3,}/g, '...');
    cleaned = cleaned.replace(/[!]{2,}/g, '!');
    cleaned = cleaned.replace(/[?]{2,}/g, '?');
    
    // Remove excessive dashes
    cleaned = cleaned.replace(/[-]{3,}/g, '--');
    
    // Remove zero-width characters
    cleaned = cleaned.replace(/[\u200B-\u200D\uFEFF]/g, '');
    
    // Remove soft hyphens
    cleaned = cleaned.replace(/\u00AD/g, '');
    
    // Normalize quotes
    cleaned = cleaned.replace(/[""]/g, '"');
    cleaned = cleaned.replace(/['']/g, "'");
    
    // Remove invisible characters
    cleaned = cleaned.replace(/[\u0080-\u009F]/g, '');
    
    return cleaned;
  }

  /**
   * Remove HTML tags from text
   */
  private removeHtmlTags(text: string): string {
    // Remove HTML tags but preserve some structure
    let cleaned = text;
    
    // Replace block elements with line breaks
    cleaned = cleaned.replace(/<(br|hr|p|div|h[1-6]|li|tr)[^>]*>/gi, '\n');
    cleaned = cleaned.replace(/<\/(p|div|h[1-6]|li|tr)[^>]*>/gi, '\n');
    
    // Remove all other HTML tags
    cleaned = cleaned.replace(/<[^>]*>/g, '');
    
    // Decode HTML entities
    cleaned = cleaned.replace(/&amp;/g, '&');
    cleaned = cleaned.replace(/&lt;/g, '<');
    cleaned = cleaned.replace(/&gt;/g, '>');
    cleaned = cleaned.replace(/&quot;/g, '"');
    cleaned = cleaned.replace(/&#39;/g, "'");
    cleaned = cleaned.replace(/&nbsp;/g, ' ');
    
    return cleaned;
  }

  /**
   * Remove URLs from text
   */
  private removeUrls(text: string): string {
    const urlRegex = /https?:\/\/[^\s]+/g;
    return text.replace(urlRegex, '');
  }

  /**
   * Remove email addresses from text
   */
  private removeEmails(text: string): string {
    const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
    return text.replace(emailRegex, '');
  }

  /**
   * Remove numbers from text
   */
  private removeNumbers(text: string): string {
    // Remove standalone numbers but preserve numbers within words
    return text.replace(/\b\d+\b/g, '');
  }

  /**
   * Truncate text to specified length while preserving word boundaries
   */
  private truncateText(text: string, maxLength: number): string {
    if (text.length <= maxLength) {
      return text;
    }

    // Try to truncate at word boundary
    const truncated = text.substring(0, maxLength);
    const lastSpace = truncated.lastIndexOf(' ');
    
    // If we can find a space in the last 20% of the text, break there
    if (lastSpace > maxLength * 0.8) {
      return truncated.substring(0, lastSpace);
    }
    
    // Otherwise, truncate at sentence boundary
    const lastSentence = truncated.lastIndexOf('.');
    if (lastSentence > maxLength * 0.6) {
      return truncated.substring(0, lastSentence + 1);
    }
    
    // If no good breaking point, truncate at character limit
    return truncated;
  }

  /**
   * Count words in text
   */
  private countWords(text: string): number {
    if (!text || text.trim().length === 0) {
      return 0;
    }
    
    // Split by whitespace and filter out empty strings
    const words = text.trim().split(/\s+/).filter(word => word.length > 0);
    return words.length;
  }

  /**
   * Generate content hash for duplicate detection
   */
  private generateHash(text: string): string {
    return crypto.createHash('sha256').update(text).digest('hex');
  }

  /**
   * Validate processed text for embedding generation
   */
  validateForEmbedding(text: string): { valid: boolean; reason?: string } {
    if (!text || typeof text !== 'string') {
      return { valid: false, reason: 'Text is not a valid string' };
    }

    const trimmed = text.trim();
    if (trimmed.length === 0) {
      return { valid: false, reason: 'Text is empty after trimming' };
    }

    if (trimmed.length > this.options.maxChunkSize) {
      return { valid: false, reason: `Text exceeds maximum chunk size of ${this.options.maxChunkSize}` };
    }

    // Check for minimum meaningful content
    const wordCount = this.countWords(trimmed);
    if (wordCount < 2) {
      return { valid: false, reason: 'Text has insufficient content for meaningful embedding' };
    }

    // Check for non-printable characters
    const nonPrintableRegex = /[\u0000-\u001F\u007F-\u009F]/g;
    if (nonPrintableRegex.test(trimmed)) {
      return { valid: false, reason: 'Text contains non-printable characters' };
    }

    return { valid: true };
  }

  /**
   * Detect duplicate content
   */
  detectDuplicateContent(texts: string[]): Map<string, string[]> {
    const hashMap = new Map<string, string[]>();
    
    for (const text of texts) {
      const result = this.preprocessText(text);
      if (!result.isEmpty) {
        const existing = hashMap.get(result.contentHash) || [];
        existing.push(text);
        hashMap.set(result.contentHash, existing);
      }
    }
    
    // Return only duplicates
    const duplicates = new Map<string, string[]>();
    for (const [hash, texts] of hashMap) {
      if (texts.length > 1) {
        duplicates.set(hash, texts);
      }
    }
    
    return duplicates;
  }

  /**
   * Get preprocessing statistics
   */
  getPreprocessingStats(results: PreprocessingResult[]): {
    totalProcessed: number;
    totalTruncated: number;
    totalEmpty: number;
    averageCharacterCount: number;
    averageWordCount: number;
    commonModifications: Map<string, number>;
  } {
    const stats = {
      totalProcessed: results.length,
      totalTruncated: results.filter(r => r.truncated).length,
      totalEmpty: results.filter(r => r.isEmpty).length,
      averageCharacterCount: 0,
      averageWordCount: 0,
      commonModifications: new Map<string, number>()
    };

    if (results.length === 0) {
      return stats;
    }

    // Calculate averages
    const validResults = results.filter(r => !r.isEmpty);
    if (validResults.length > 0) {
      stats.averageCharacterCount = validResults.reduce((sum, r) => sum + r.characterCount, 0) / validResults.length;
      stats.averageWordCount = validResults.reduce((sum, r) => sum + r.wordCount, 0) / validResults.length;
    }

    // Count modification frequency
    for (const result of results) {
      for (const modification of result.modifications) {
        const count = stats.commonModifications.get(modification) || 0;
        stats.commonModifications.set(modification, count + 1);
      }
    }

    return stats;
  }

  /**
   * Update preprocessing options
   */
  updateOptions(newOptions: Partial<TextPreprocessingOptions>): void {
    this.options = { ...this.options, ...newOptions };
  }

  /**
   * Get current preprocessing options
   */
  getOptions(): TextPreprocessingOptions {
    return { ...this.options };
  }
}

// Export singleton instance
export const textPreprocessor = new TextPreprocessor();
export default textPreprocessor;
