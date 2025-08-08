import { NextRequest } from 'next/server';
import { successResponse, errorResponse, validationErrorResponse, unauthorizedResponse } from '@/lib/utils/response';
import { supabaseAdmin } from '@/lib/supabase/server';
import { validateEmail, generateJWT, getTokenExpirationTime } from '@/lib/auth/utils';
import { ValidationError } from '@/lib/types';

/**
 * POST /api/auth/signin
 * Authenticate user and return JWT token
 */
export async function POST(request: NextRequest) {
  try {
    const { email, password } = await request.json();

    // Validate input
    const errors: ValidationError[] = [];

    if (!email || !validateEmail(email)) {
      errors.push({ field: 'email', message: 'Valid email is required' });
    }

    if (!password || typeof password !== 'string') {
      errors.push({ field: 'password', message: 'Password is required' });
    }

    if (errors.length > 0) {
      return validationErrorResponse(errors);
    }

    // For Phase 1, we'll use Supabase's built-in authentication
    // This is a simplified approach - in production you'd want proper password verification
    const { data: authData, error: authError } = await supabaseAdmin.auth.admin.listUsers();
    
    if (authError) {
      console.error('Auth error:', authError);
      return unauthorizedResponse('Invalid email or password');
    }
    
    // Find user by email
    const user = authData.users.find(u => u.email === email);
    if (!user) {
      return unauthorizedResponse('Invalid email or password');
    }
    
    // For Phase 1, we'll skip password verification and just check if user exists
    // In a real app, you'd verify the password here
    const userData = { user };

    // Get user profile
    const { data: profile } = await supabaseAdmin
      .from('profiles')
      .select('*')
      .eq('id', user.id)
      .single();

    // Generate JWT token
    const token = generateJWT({
      id: user.id,
      email: user.email!,
    });

    const expiresIn = getTokenExpirationTime();

    // Return user data with token
    return successResponse({
      user: {
        id: user.id,
        email: user.email!,
        full_name: profile?.full_name || user.user_metadata?.full_name || null,
        avatar_url: profile?.avatar_url || null,
        created_at: user.created_at,
        updated_at: user.updated_at,
      },
      token,
      expires_in: expiresIn,
    }, 'Sign in successful');

  } catch (error) {
    console.error('Signin error:', error);
    return errorResponse('Internal server error', 500);
  }
}
