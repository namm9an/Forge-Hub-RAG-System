export interface DocumentChunk {
  id?: string;
  document_id?: string;
  chunk_index: number;
  content: string;
  metadata: ChunkMetadata;
}

export interface ChunkMetadata {
  start_char: number;
  end_char: number;
  chunk_type: 'paragraph' | 'list' | 'heading' | 'table' | 'mixed';
  section_title?: string;
  source_page?: number;
  word_count: number;
  char_count: number;
  has_code?: boolean;
  language?: string;
}

export interface ChunkingOptions {
  chunkSize?: number;
  chunkOverlap?: number;
  preserveParagraphs?: boolean;
  preserveCodeBlocks?: boolean;
  minChunkSize?: number;
}

const DEFAULT_OPTIONS: Required<ChunkingOptions> = {
  chunkSize: 1000,
  chunkOverlap: 200,
  preserveParagraphs: true,
  preserveCodeBlocks: true,
  minChunkSize: 100,
};

/**
 * Split text into semantic chunks with overlap
 * @param text - The text to chunk
 * @param metadata - Additional metadata about the document
 * @param options - Chunking options
 * @returns Array of document chunks
 */
export function chunkText(
  text: string,
  metadata: Record<string, any> = {},
  options: ChunkingOptions = {}
): DocumentChunk[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Normalize text
  const normalizedText = normalizeText(text);
  
  // Extract structural elements
  const sections = extractSections(normalizedText);
  
  // Create chunks
  const chunks: DocumentChunk[] = [];
  let globalChunkIndex = 0;
  
  for (const section of sections) {
    const sectionChunks = createSemanticChunks(
      section.content,
      opts,
      globalChunkIndex,
      section.title
    );
    
    chunks.push(...sectionChunks);
    globalChunkIndex += sectionChunks.length;
  }
  
  // Add overlap between chunks
  const chunksWithOverlap = addOverlapToChunks(chunks, normalizedText, opts.chunkOverlap);
  
  // Enrich metadata
  return chunksWithOverlap.map((chunk, index) => ({
    ...chunk,
    chunk_index: index,
    metadata: {
      ...chunk.metadata,
      ...extractChunkMetadata(chunk.content, index),
    },
  }));
}

/**
 * Create semantic chunks from text
 */
export function createSemanticChunks(
  text: string,
  options: Required<ChunkingOptions>,
  startIndex: number = 0,
  sectionTitle?: string
): DocumentChunk[] {
  const chunks: DocumentChunk[] = [];
  
  // Split into paragraphs first
  const paragraphs = splitIntoParagraphs(text);
  
  let currentChunk = '';
  let currentStartChar = 0;
  let paragraphsInChunk: string[] = [];
  
  for (let i = 0; i < paragraphs.length; i++) {
    const paragraph = paragraphs[i];
    const paragraphLength = paragraph.length;
    
    // Check if this is a code block
    const isCodeBlock = detectCodeBlock(paragraph);
    
    // If preserving code blocks and this is one, make it its own chunk
    if (options.preserveCodeBlocks && isCodeBlock && currentChunk) {
      // Save current chunk first
      if (currentChunk.trim().length >= options.minChunkSize) {
        chunks.push(createChunk(
          currentChunk,
          startIndex + chunks.length,
          currentStartChar,
          currentStartChar + currentChunk.length,
          detectChunkType(paragraphsInChunk),
          sectionTitle
        ));
      }
      
      // Create code block chunk
      chunks.push(createChunk(
        paragraph,
        startIndex + chunks.length,
        currentStartChar + currentChunk.length,
        currentStartChar + currentChunk.length + paragraphLength,
        'paragraph',
        sectionTitle,
        true
      ));
      
      // Reset for next chunk
      currentChunk = '';
      currentStartChar += currentChunk.length + paragraphLength;
      paragraphsInChunk = [];
      continue;
    }
    
    // Check if adding this paragraph would exceed chunk size
    if (currentChunk && (currentChunk.length + paragraphLength > options.chunkSize)) {
      // Save current chunk
      if (currentChunk.trim().length >= options.minChunkSize) {
        chunks.push(createChunk(
          currentChunk,
          startIndex + chunks.length,
          currentStartChar,
          currentStartChar + currentChunk.length,
          detectChunkType(paragraphsInChunk),
          sectionTitle
        ));
      }
      
      // Start new chunk
      currentStartChar += currentChunk.length;
      currentChunk = paragraph;
      paragraphsInChunk = [paragraph];
    } else {
      // Add to current chunk
      if (currentChunk) {
        currentChunk += '\n\n' + paragraph;
      } else {
        currentChunk = paragraph;
      }
      paragraphsInChunk.push(paragraph);
    }
  }
  
  // Add final chunk
  if (currentChunk.trim().length >= options.minChunkSize) {
    chunks.push(createChunk(
      currentChunk,
      startIndex + chunks.length,
      currentStartChar,
      currentStartChar + currentChunk.length,
      detectChunkType(paragraphsInChunk),
      sectionTitle
    ));
  }
  
  return chunks;
}

/**
 * Extract metadata from chunk content
 */
export function extractMetadata(text: string, chunkIndex: number): ChunkMetadata {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const wordCount = words.length;
  const charCount = text.length;
  
  return {
    start_char: 0,
    end_char: charCount,
    chunk_type: detectChunkType([text]),
    word_count: wordCount,
    char_count: charCount,
    has_code: detectCodeBlock(text),
  };
}

// Helper functions

function normalizeText(text: string): string {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function extractSections(text: string): Array<{ title?: string; content: string }> {
  // Simple section extraction based on markdown headers
  const sections: Array<{ title?: string; content: string }> = [];
  const lines = text.split('\n');
  
  let currentSection: { title?: string; content: string } = { content: '' };
  let inSection = false;
  
  for (const line of lines) {
    const headerMatch = line.match(/^#{1,6}\s+(.+)$/);
    
    if (headerMatch) {
      // Save previous section if exists
      if (currentSection.content.trim()) {
        sections.push(currentSection);
      }
      
      // Start new section
      currentSection = {
        title: headerMatch[1].trim(),
        content: '',
      };
      inSection = true;
    } else {
      currentSection.content += (currentSection.content ? '\n' : '') + line;
    }
  }
  
  // Add final section
  if (currentSection.content.trim()) {
    sections.push(currentSection);
  }
  
  // If no sections were found, treat entire text as one section
  if (sections.length === 0) {
    sections.push({ content: text });
  }
  
  return sections;
}

function splitIntoParagraphs(text: string): string[] {
  // Split by double newlines, but preserve code blocks
  const paragraphs = text.split(/\n\n+/);
  
  // Merge code blocks that might have been split
  const mergedParagraphs: string[] = [];
  let inCodeBlock = false;
  let codeBlockContent = '';
  
  for (const paragraph of paragraphs) {
    if (paragraph.startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeBlockContent = paragraph;
      } else {
        codeBlockContent += '\n\n' + paragraph;
        if (paragraph.endsWith('```')) {
          inCodeBlock = false;
          mergedParagraphs.push(codeBlockContent);
          codeBlockContent = '';
        }
      }
    } else if (inCodeBlock) {
      codeBlockContent += '\n\n' + paragraph;
    } else {
      mergedParagraphs.push(paragraph);
    }
  }
  
  // Add any remaining code block content
  if (codeBlockContent) {
    mergedParagraphs.push(codeBlockContent);
  }
  
  return mergedParagraphs.filter(p => p.trim().length > 0);
}

function detectCodeBlock(text: string): boolean {
  // Check for markdown code blocks
  if (text.includes('```')) return true;
  
  // Check for indented code (4 spaces or tab at start of multiple lines)
  const lines = text.split('\n');
  const indentedLines = lines.filter(line => /^(\s{4}|\t)/.test(line));
  
  // If more than 50% of lines are indented, likely a code block
  return indentedLines.length > lines.length * 0.5;
}

function detectChunkType(paragraphs: string[]): ChunkMetadata['chunk_type'] {
  if (paragraphs.length === 0) return 'paragraph';
  
  const types = new Set<string>();
  
  for (const paragraph of paragraphs) {
    if (/^#{1,6}\s/.test(paragraph)) {
      types.add('heading');
    } else if (/^[\*\-\+]\s/.test(paragraph) || /^\d+\.\s/.test(paragraph)) {
      types.add('list');
    } else if (/\|.*\|/.test(paragraph) && paragraph.split('\n').length > 2) {
      types.add('table');
    } else {
      types.add('paragraph');
    }
  }
  
  if (types.size === 1) {
    return Array.from(types)[0] as ChunkMetadata['chunk_type'];
  }
  
  return 'mixed';
}

function createChunk(
  content: string,
  index: number,
  startChar: number,
  endChar: number,
  chunkType: ChunkMetadata['chunk_type'],
  sectionTitle?: string,
  hasCode?: boolean
): DocumentChunk {
  const words = content.split(/\s+/).filter(w => w.length > 0);
  
  return {
    chunk_index: index,
    content: content.trim(),
    metadata: {
      start_char: startChar,
      end_char: endChar,
      chunk_type: chunkType,
      section_title: sectionTitle,
      word_count: words.length,
      char_count: content.length,
      has_code: hasCode || detectCodeBlock(content),
    },
  };
}

function addOverlapToChunks(
  chunks: DocumentChunk[],
  originalText: string,
  overlapSize: number
): DocumentChunk[] {
  if (chunks.length <= 1 || overlapSize <= 0) return chunks;
  
  const enhancedChunks: DocumentChunk[] = [];
  
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    let enhancedContent = chunk.content;
    
    // Add overlap from previous chunk
    if (i > 0) {
      const prevChunk = chunks[i - 1];
      const overlapText = getOverlapText(prevChunk.content, overlapSize, 'end');
      if (overlapText) {
        enhancedContent = overlapText + '\n\n' + enhancedContent;
      }
    }
    
    // Add overlap from next chunk
    if (i < chunks.length - 1) {
      const nextChunk = chunks[i + 1];
      const overlapText = getOverlapText(nextChunk.content, overlapSize, 'start');
      if (overlapText) {
        enhancedContent = enhancedContent + '\n\n' + overlapText;
      }
    }
    
    enhancedChunks.push({
      ...chunk,
      content: enhancedContent,
      metadata: {
        ...chunk.metadata,
        char_count: enhancedContent.length,
        word_count: enhancedContent.split(/\s+/).filter(w => w.length > 0).length,
      },
    });
  }
  
  return enhancedChunks;
}

function getOverlapText(text: string, size: number, position: 'start' | 'end'): string {
  if (text.length <= size) return text;
  
  if (position === 'start') {
    // Get first N characters, trying to break at sentence boundary
    let overlap = text.substring(0, size);
    const lastPeriod = overlap.lastIndexOf('.');
    const lastNewline = overlap.lastIndexOf('\n');
    
    const breakPoint = Math.max(lastPeriod, lastNewline);
    if (breakPoint > size * 0.5) {
      overlap = overlap.substring(0, breakPoint + 1);
    }
    
    return overlap.trim();
  } else {
    // Get last N characters, trying to break at sentence boundary
    let overlap = text.substring(text.length - size);
    const firstPeriod = overlap.indexOf('.');
    const firstNewline = overlap.indexOf('\n');
    
    const breakPoint = firstPeriod >= 0 && firstNewline >= 0 
      ? Math.min(firstPeriod, firstNewline)
      : Math.max(firstPeriod, firstNewline);
      
    if (breakPoint >= 0 && breakPoint < size * 0.5) {
      overlap = overlap.substring(breakPoint + 1);
    }
    
    return overlap.trim();
  }
}

function extractChunkMetadata(content: string, index: number): Partial<ChunkMetadata> {
  // Detect programming language if code is present
  let language: string | undefined;
  
  const codeBlockMatch = content.match(/```(\w+)/);
  if (codeBlockMatch) {
    language = codeBlockMatch[1];
  }
  
  return {
    language,
  };
}
