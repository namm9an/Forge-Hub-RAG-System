import { supabase } from '@/lib/supabase/client';
import { supabaseAdmin } from '@/lib/supabase/server';
import { Database } from '@/lib/supabase/database.types';

export type SupabaseClient = typeof supabase;
export type SupabaseAdminClient = typeof supabaseAdmin;

/**
 * Test database connection
 * @returns Promise<boolean> - True if connection is successful
 */
export async function testDatabaseConnection(): Promise<boolean> {
  try {
    // Test database connection with a simple query
    const { error } = await supabaseAdmin
      .from('profiles')
      .select('*')
      .limit(1);

    // If there's an error, connection might be ok but table doesn't exist
    if (error && error.message.includes('does not exist')) {
      console.warn('Database connected but tables not created yet');
      return true; // Connection is fine, just need to run schema
    }

    if (error) {
      console.error('Database connection test failed:', error);
      return false;
    }

    return true;
  } catch (error) {
    console.error('Database connection test failed:', error);
    return false;
  }
}

/**
 * Get database client based on context
 * @param useAdmin - Whether to use admin client
 * @returns Supabase client instance
 */
export function getDatabaseClient(useAdmin: boolean = false): SupabaseClient | SupabaseAdminClient {
  return useAdmin ? supabaseAdmin : supabase;
}

/**
 * Execute a raw SQL query using the admin client
 * @param query - SQL query string
 * @param params - Query parameters
 * @returns Promise with query result
 */
export async function executeRawQuery(
  query: string,
  params?: any[]
): Promise<{ data: any; error: any }> {
  try {
    const { data, error } = await supabaseAdmin.rpc('execute_sql', {
      query,
      params: params || [],
    });

    return { data, error };
  } catch (error) {
    console.error('Raw query execution failed:', error);
    return { data: null, error };
  }
}

/**
 * Execute a database transaction
 * @param callback - Function to execute within transaction
 * @returns Promise with transaction result
 */
export async function withTransaction<T>(
  callback: (client: SupabaseAdminClient) => Promise<T>
): Promise<T> {
  try {
    const result = await callback(supabaseAdmin);
    return result;
  } catch (error) {
    console.error('Transaction failed:', error);
    throw error;
  }
}

/**
 * Get table record count
 * @param tableName - Name of the table
 * @param conditions - Optional where conditions
 * @returns Promise<number> - Record count
 */
export async function getTableCount(
  tableName: string,
  conditions?: Record<string, any>
): Promise<number> {
  try {
    let query = supabaseAdmin
      .from(tableName)
      .select('*', { count: 'exact', head: true });

    if (conditions) {
      Object.entries(conditions).forEach(([key, value]) => {
        query = query.eq(key, value);
      });
    }

    const { count, error } = await query;

    if (error) {
      console.error('Error getting table count:', error);
      return 0;
    }

    return count || 0;
  } catch (error) {
    console.error('Error getting table count:', error);
    return 0;
  }
}

/**
 * Check if a record exists
 * @param tableName - Name of the table
 * @param conditions - Where conditions
 * @returns Promise<boolean> - True if record exists
 */
export async function recordExists(
  tableName: string,
  conditions: Record<string, any>
): Promise<boolean> {
  try {
    let query = supabaseAdmin
      .from(tableName)
      .select('id', { count: 'exact', head: true });

    Object.entries(conditions).forEach(([key, value]) => {
      query = query.eq(key, value);
    });

    const { count, error } = await query;

    if (error) {
      console.error('Error checking record existence:', error);
      return false;
    }

    return (count || 0) > 0;
  } catch (error) {
    console.error('Error checking record existence:', error);
    return false;
  }
}

/**
 * Get user profile by ID
 * @param userId - User ID
 * @returns Promise with user profile
 */
export async function getUserProfile(userId: string) {
  try {
    const { data, error } = await supabaseAdmin
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single();

    if (error) {
      console.error('Error fetching user profile:', error);
      return null;
    }

    return data;
  } catch (error) {
    console.error('Error fetching user profile:', error);
    return null;
  }
}

/**
 * Create or update user profile
 * @param userId - User ID
 * @param profileData - Profile data
 * @returns Promise with created/updated profile
 */
export async function upsertUserProfile(
  userId: string,
  profileData: Partial<Database['public']['Tables']['profiles']['Insert']>
) {
  try {
    const { data, error } = await supabaseAdmin
      .from('profiles')
      .upsert({
        id: userId,
        ...profileData,
        updated_at: new Date().toISOString(),
      })
      .select()
      .single();

    if (error) {
      console.error('Error upserting user profile:', error);
      throw error;
    }

    return data;
  } catch (error) {
    console.error('Error upserting user profile:', error);
    throw error;
  }
}

/**
 * Delete user and all associated data
 * @param userId - User ID
 * @returns Promise<boolean> - True if deletion was successful
 */
export async function deleteUserData(userId: string): Promise<boolean> {
  try {
    // Delete in cascade order due to foreign key constraints
    const operations = [
      supabaseAdmin.from('search_history').delete().eq('user_id', userId),
      supabaseAdmin.from('embeddings').delete().in('chunk_id', 
        supabaseAdmin.from('document_chunks').select('id').in('document_id',
          supabaseAdmin.from('documents').select('id').eq('user_id', userId)
        )
      ),
      supabaseAdmin.from('document_chunks').delete().in('document_id',
        supabaseAdmin.from('documents').select('id').eq('user_id', userId)
      ),
      supabaseAdmin.from('documents').delete().eq('user_id', userId),
      supabaseAdmin.from('profiles').delete().eq('id', userId),
    ];

    for (const operation of operations) {
      const { error } = await operation;
      if (error) {
        console.error('Error deleting user data:', error);
        return false;
      }
    }

    return true;
  } catch (error) {
    console.error('Error deleting user data:', error);
    return false;
  }
}

/**
 * Get database statistics
 * @returns Promise with database statistics
 */
export async function getDatabaseStats() {
  try {
    const [
      profilesCount,
      documentsCount,
      chunksCount,
      embeddingsCount,
      searchHistoryCount,
    ] = await Promise.all([
      getTableCount('profiles'),
      getTableCount('documents'),
      getTableCount('document_chunks'),
      getTableCount('embeddings'),
      getTableCount('search_history'),
    ]);

    return {
      profiles: profilesCount,
      documents: documentsCount,
      chunks: chunksCount,
      embeddings: embeddingsCount,
      searchHistory: searchHistoryCount,
      total: profilesCount + documentsCount + chunksCount + embeddingsCount + searchHistoryCount,
    };
  } catch (error) {
    console.error('Error getting database stats:', error);
    return {
      profiles: 0,
      documents: 0,
      chunks: 0,
      embeddings: 0,
      searchHistory: 0,
      total: 0,
    };
  }
}

/**
 * Clean up old records
 * @param tableName - Name of the table
 * @param dateColumn - Date column to filter by
 * @param daysOld - Number of days to keep
 * @returns Promise<number> - Number of deleted records
 */
export async function cleanupOldRecords(
  tableName: string,
  dateColumn: string = 'created_at',
  daysOld: number = 30
): Promise<number> {
  try {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - daysOld);

    const { data, error } = await supabaseAdmin
      .from(tableName)
      .delete()
      .lt(dateColumn, cutoffDate.toISOString())
      .select('id');

    if (error) {
      console.error('Error cleaning up old records:', error);
      return 0;
    }

    return data?.length || 0;
  } catch (error) {
    console.error('Error cleaning up old records:', error);
    return 0;
  }
}
