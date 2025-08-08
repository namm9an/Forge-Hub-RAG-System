import { GoogleGenerativeAI } from '@google/generative-ai';

const apiKey = process.env.GEMINI_API_KEY;
const model = process.env.GEMINI_MODEL || 'gemini-2.0-flash-exp';
const embeddingModel = process.env.GEMINI_EMBEDDING_MODEL || 'text-embedding-004';

if (!apiKey) {
  console.warn('GEMINI_API_KEY environment variable is not set');
}

const genAI = apiKey ? new GoogleGenerativeAI(apiKey) : null;

export class GeminiClient {
  private generativeModel;
  private embeddingModel;

  constructor() {
    if (!genAI) {
      throw new Error('Gemini API is not available - missing API key');
    }
    this.generativeModel = genAI.getGenerativeModel({ model });
    this.embeddingModel = genAI.getGenerativeModel({ model: embeddingModel });
  }

  /**
   * Generate text embedding for the given text
   * @param text - Text to generate embedding for
   * @returns Promise<number[]> - Array of embedding values
   */
  async generateEmbedding(text: string): Promise<number[]> {
    try {
      if (!text || text.trim().length === 0) {
        throw new Error('Text cannot be empty');
      }

      const result = await this.embeddingModel.embedContent(text);
      
      if (!result.embedding || !result.embedding.values) {
        throw new Error('Failed to generate embedding');
      }

      return result.embedding.values;
    } catch (error) {
      console.error('Error generating embedding:', error);
      throw new Error(`Failed to generate embedding: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Generate completion for the given prompt
   * @param prompt - Prompt to generate completion for
   * @returns Promise<string> - Generated completion text
   */
  async generateCompletion(prompt: string): Promise<string> {
    try {
      if (!prompt || prompt.trim().length === 0) {
        throw new Error('Prompt cannot be empty');
      }

      const result = await this.generativeModel.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      if (!text) {
        throw new Error('No response generated');
      }

      return text;
    } catch (error) {
      console.error('Error generating completion:', error);
      throw new Error(`Failed to generate completion: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Generate streaming completion for the given prompt
   * @param prompt - Prompt to generate completion for
   * @returns AsyncGenerator<string> - Streaming completion text
   */
  async* generateStreamingCompletion(prompt: string): AsyncGenerator<string> {
    try {
      if (!prompt || prompt.trim().length === 0) {
        throw new Error('Prompt cannot be empty');
      }

      const result = await this.generativeModel.generateContentStream(prompt);
      
      for await (const chunk of result.stream) {
        const chunkText = chunk.text();
        if (chunkText) {
          yield chunkText;
        }
      }
    } catch (error) {
      console.error('Error generating streaming completion:', error);
      throw new Error(`Failed to generate streaming completion: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Test the connection to Gemini API
   * @returns Promise<boolean> - True if connection is successful
   */
  async testConnection(): Promise<boolean> {
    try {
      const testPrompt = 'Hello, this is a test.';
      await this.generateCompletion(testPrompt);
      return true;
    } catch (error) {
      console.error('Gemini API connection test failed:', error);
      return false;
    }
  }

  /**
   * Get model information
   * @returns Object containing model configuration
   */
  getModelInfo() {
    return {
      model,
      embeddingModel,
      embeddingDimensions: 768,
      apiAvailable: !!apiKey,
    };
  }
}

// Export singleton instance with error handling
let geminiClient: GeminiClient | null = null;

try {
  geminiClient = new GeminiClient();
} catch (error) {
  console.warn('Gemini client initialization failed:', error);
  geminiClient = null;
}

export { geminiClient };
export default geminiClient;
