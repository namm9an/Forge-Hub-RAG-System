import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse } from '@/lib/utils/response';
import { supabaseAdmin } from '@/lib/supabase/server';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';

/**
 * GET /api/auth/me
 * Get profile of authenticated user
 */
export async function GET(request: NextRequest) {
  try {
    const token = extractTokenFromHeader(request.headers.get('authorization'));
    if (!token) {
      return unauthorizedResponse('Authorization token is required');
    }

    // Verify JWT token
    const decoded = verifyJWT(token);
    if (!decoded.userId) {
      return unauthorizedResponse('Invalid token');
    }

    // Get user from Supabase Auth
    const { data: authUser, error: authError } = await supabaseAdmin.auth.admin.getUserById(decoded.userId);
    
    if (authError || !authUser.user) {
      console.error('Auth user fetch error:', authError);
      return unauthorizedResponse('User not found');
    }

    // Get user profile
    const { data: profile, error: profileError } = await supabaseAdmin
      .from('profiles')
      .select('*')
      .eq('id', decoded.userId)
      .single();

    if (profileError) {
      console.error('Profile fetch error:', profileError);
    }

    // Return user data
    return successResponse({
      user: {
        id: authUser.user.id,
        email: authUser.user.email!,
        full_name: profile?.full_name || authUser.user.user_metadata?.full_name || null,
        avatar_url: profile?.avatar_url || null,
        created_at: authUser.user.created_at,
        updated_at: profile?.updated_at || authUser.user.updated_at,
      },
    });

  } catch (error) {
    console.error('Get profile error:', error);
    
    if (error instanceof Error) {
      if (error.message.includes('expired')) {
        return unauthorizedResponse('Token has expired');
      }
      if (error.message.includes('invalid')) {
        return unauthorizedResponse('Invalid token');
      }
    }
    
    return errorResponse('Internal server error', 500);
  }
}
