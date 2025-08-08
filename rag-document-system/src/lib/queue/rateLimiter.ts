import { createSupabaseClient } from '../supabase/client';
import { RateLimitTracking } from '../types';

export class RateLimiter {
  private supabase: ReturnType<typeof createSupabaseClient>;
  private limits: Map<string, { requests: number; tokens: number; window: number }>;

  constructor() {
    this.supabase = createSupabaseClient();
    this.limits = new Map([
      ['gemini_embeddings', { requests: 15, tokens: 32000, window: 60000 }], // per minute
      ['similarity_search', { requests: 60, tokens: 0, window: 60000 }], // per minute
      ['batch_processing', { requests: 5, tokens: 0, window: 60000 }] // per minute
    ]);
  }

  /**
   * Check if rate limit is exceeded
   */
  async checkRateLimit(serviceName: string, userId?: string): Promise<boolean> {
    try {
      const limit = this.limits.get(serviceName);
      if (!limit) {
        return true; // No limit configured, allow
      }

      const now = new Date();
      const windowStart = new Date(now.getTime() - limit.window);

      // Get current usage within window
      const { data: usage, error } = await this.supabase
        .from('rate_limit_tracking')
        .select('requests_count, tokens_used')
        .eq('service_name', serviceName)
        .eq('user_id', userId || null)
        .gte('reset_time', windowStart.toISOString())
        .single();

      if (error && error.code !== 'PGRST116') {
        console.error('Error checking rate limit:', error);
        return true; // Allow on error
      }

      if (!usage) {
        return true; // No usage recorded, allow
      }

      // Check if limits are exceeded
      const requestsExceeded = usage.requests_count >= limit.requests;
      const tokensExceeded = limit.tokens > 0 && usage.tokens_used >= limit.tokens;

      return !(requestsExceeded || tokensExceeded);

    } catch (error) {
      console.error('Error in rate limit check:', error);
      return true; // Allow on error
    }
  }

  /**
   * Wait for rate limit reset
   */
  async waitForRateLimit(serviceName: string, userId?: string): Promise<void> {
    const limit = this.limits.get(serviceName);
    if (!limit) {
      return;
    }

    const now = new Date();
    const windowStart = new Date(now.getTime() - limit.window);

    // Get the oldest request in current window
    const { data: oldestRequest, error } = await this.supabase
      .from('rate_limit_tracking')
      .select('reset_time')
      .eq('service_name', serviceName)
      .eq('user_id', userId || null)
      .gte('reset_time', windowStart.toISOString())
      .order('reset_time', { ascending: true })
      .limit(1)
      .single();

    if (error || !oldestRequest) {
      return;
    }

    const resetTime = new Date(oldestRequest.reset_time);
    const waitTime = Math.max(0, resetTime.getTime() + limit.window - now.getTime());

    if (waitTime > 0) {
      console.log(`Rate limit exceeded for ${serviceName}, waiting ${waitTime}ms`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
  }

  /**
   * Update rate limit statistics
   */
  async updateRateLimitStats(
    serviceName: string,
    requestCount: number = 1,
    tokensUsed: number = 0,
    userId?: string
  ): Promise<void> {
    try {
      const now = new Date();
      const resetTime = new Date(now.getTime() + (this.limits.get(serviceName)?.window || 60000));

      // Try to update existing record
      const { data: existing, error: selectError } = await this.supabase
        .from('rate_limit_tracking')
        .select('id, requests_count, tokens_used')
        .eq('service_name', serviceName)
        .eq('user_id', userId || null)
        .gte('reset_time', now.toISOString())
        .single();

      if (selectError && selectError.code !== 'PGRST116') {
        console.error('Error selecting rate limit record:', selectError);
        return;
      }

      if (existing) {
        // Update existing record
        await this.supabase
          .from('rate_limit_tracking')
          .update({
            requests_count: existing.requests_count + requestCount,
            tokens_used: existing.tokens_used + tokensUsed,
            updated_at: now.toISOString()
          })
          .eq('id', existing.id);
      } else {
        // Create new record
        await this.supabase
          .from('rate_limit_tracking')
          .insert({
            service_name: serviceName,
            user_id: userId || null,
            requests_count: requestCount,
            tokens_used: tokensUsed,
            reset_time: resetTime.toISOString()
          });
      }

    } catch (error) {
      console.error('Error updating rate limit stats:', error);
    }
  }

  /**
   * Get remaining quota
   */
  async getRemainingQuota(serviceName: string, userId?: string): Promise<{
    requests_remaining: number;
    tokens_remaining: number;
    reset_time: string;
  }> {
    try {
      const limit = this.limits.get(serviceName);
      if (!limit) {
        return {
          requests_remaining: Infinity,
          tokens_remaining: Infinity,
          reset_time: new Date().toISOString()
        };
      }

      const now = new Date();
      const windowStart = new Date(now.getTime() - limit.window);

      const { data: usage, error } = await this.supabase
        .from('rate_limit_tracking')
        .select('requests_count, tokens_used, reset_time')
        .eq('service_name', serviceName)
        .eq('user_id', userId || null)
        .gte('reset_time', windowStart.toISOString())
        .order('reset_time', { ascending: false })
        .limit(1)
        .single();

      if (error || !usage) {
        return {
          requests_remaining: limit.requests,
          tokens_remaining: limit.tokens,
          reset_time: new Date(now.getTime() + limit.window).toISOString()
        };
      }

      return {
        requests_remaining: Math.max(0, limit.requests - usage.requests_count),
        tokens_remaining: limit.tokens > 0 ? Math.max(0, limit.tokens - usage.tokens_used) : Infinity,
        reset_time: usage.reset_time
      };

    } catch (error) {
      console.error('Error getting remaining quota:', error);
      const limit = this.limits.get(serviceName);
      return {
        requests_remaining: limit?.requests || Infinity,
        tokens_remaining: limit?.tokens || Infinity,
        reset_time: new Date().toISOString()
      };
    }
  }

  /**
   * Reset rate limits for a service
   */
  async resetRateLimits(serviceName: string, userId?: string): Promise<void> {
    try {
      await this.supabase
        .from('rate_limit_tracking')
        .delete()
        .eq('service_name', serviceName)
        .eq('user_id', userId || null);

      console.log(`Rate limits reset for ${serviceName}${userId ? ` (user: ${userId})` : ''}`);
    } catch (error) {
      console.error('Error resetting rate limits:', error);
    }
  }

  /**
   * Get rate limit configuration
   */
  getRateLimitConfig(): Map<string, { requests: number; tokens: number; window: number }> {
    return new Map(this.limits);
  }

  /**
   * Update rate limit configuration
   */
  updateRateLimitConfig(
    serviceName: string,
    config: { requests: number; tokens: number; window: number }
  ): void {
    this.limits.set(serviceName, config);
  }

  /**
   * Clean up old rate limit records
   */
  async cleanupOldRecords(): Promise<number> {
    try {
      const cutoffTime = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours ago

      const { data, error } = await this.supabase
        .from('rate_limit_tracking')
        .delete()
        .lt('reset_time', cutoffTime.toISOString())
        .select('id');

      if (error) {
        console.error('Error cleaning up old rate limit records:', error);
        return 0;
      }

      return data?.length || 0;
    } catch (error) {
      console.error('Error in cleanup:', error);
      return 0;
    }
  }
}

// Export singleton instance
export const rateLimiter = new RateLimiter();
export default rateLimiter;
