import { NextRequest } from 'next/server';
import { successResponse, errorResponse } from '@/lib/utils/response';
import { getDatabaseStats } from '@/lib/utils/database';
import { geminiClient } from '@/lib/gemini/client';

/**
 * GET /api/system/info
 * Get system information
 */
export async function GET(request: NextRequest) {
  try {
    const startTime = Date.now();
    
    // Get database statistics
    const dbStats = await getDatabaseStats();
    
    // Get Gemini model information
    const geminiInfo = geminiClient ? geminiClient.getModelInfo() : {
      model: 'gemini-2.0-flash-exp',
      embeddingModel: 'text-embedding-004',
      embeddingDimensions: 768,
      apiAvailable: false,
    };
    
    const responseTime = Date.now() - startTime;
    
    const systemInfo = {
      version: '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      uptime: process.uptime(),
      responseTime,
      timestamp: new Date().toISOString(),
      
      // Database statistics
      database: {
        connected: true,
        statistics: dbStats,
      },
      
      // AI service information
      ai: {
        provider: 'Google Gemini',
        model: geminiInfo.model,
        embeddingModel: geminiInfo.embeddingModel,
        embeddingDimensions: geminiInfo.embeddingDimensions,
        available: geminiInfo.apiAvailable,
      },
      
      // System limits
      limits: {
        maxFileSize: 10 * 1024 * 1024, // 10MB
        maxDocuments: 100, // Per user
        allowedFileTypes: ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        rateLimit: {
          global: {
            limit: 100,
            window: 15 * 60 * 1000, // 15 minutes
          },
          auth: {
            limit: 5,
            window: 60 * 1000, // 1 minute
          },
          upload: {
            limit: 10,
            window: 60 * 60 * 1000, // 1 hour
          },
        },
      },
      
      // Feature flags
      features: {
        documentUpload: true,
        documentProcessing: false, // Phase 2
        vectorSearch: false, // Phase 3
        aiChat: false, // Phase 4
        realTimeSync: false, // Future
      },
      
      // Service health
      health: {
        status: 'healthy',
        checks: {
          database: 'passing',
          gemini: geminiInfo.apiAvailable ? 'passing' : 'warning',
          memory: process.memoryUsage().heapUsed < 500 * 1024 * 1024 ? 'passing' : 'warning',
        },
      },
    };
    
    return successResponse(systemInfo);
    
  } catch (error) {
    console.error('System info error:', error);
    return errorResponse('Failed to get system information', 500);
  }
}
