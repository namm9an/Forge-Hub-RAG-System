import { NextRequest } from 'next/server';
import { successResponse, errorResponse } from '@/lib/utils/response';
import { testDatabaseConnection } from '@/lib/utils/database';
import { geminiClient } from '@/lib/gemini/client';

/**
 * Health check endpoint
 * GET /api/health
 */
export async function GET(request: NextRequest) {
  try {
    const startTime = Date.now();
    
    // Test database connection
    const isDatabaseConnected = await testDatabaseConnection();
    
    // Test Gemini API connection
    let isGeminiConnected = false;
    try {
      if (geminiClient) {
        isGeminiConnected = await geminiClient.testConnection();
      }
    } catch (error) {
      console.warn('Gemini API connection test failed:', error);
    }
    
    const responseTime = Date.now() - startTime;
    
    const healthData = {
      status: isDatabaseConnected ? 'ok' : 'error',
      timestamp: new Date().toISOString(),
      services: {
        database: isDatabaseConnected ? 'connected' : 'disconnected',
        gemini: isGeminiConnected ? 'connected' : 'disconnected',
      },
      uptime: process.uptime(),
      responseTime,
      environment: process.env.NODE_ENV || 'development',
      version: '1.0.0',
    };
    
    if (!isDatabaseConnected) {
      return errorResponse('Database connection failed', 503);
    }
    
    return successResponse(healthData);
  } catch (error) {
    console.error('Health check failed:', error);
    return errorResponse('Health check failed', 500);
  }
}
