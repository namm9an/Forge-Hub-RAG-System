import { NextRequest } from 'next/server';
import { successResponse, errorResponse, validationErrorResponse } from '@/lib/utils/response';
import { supabaseAdmin } from '@/lib/supabase/server';
import { validateEmail, validatePassword, generateJWT, getTokenExpirationTime } from '@/lib/auth/utils';
import { ValidationError } from '@/lib/types';

/**
 * POST /api/auth/signup
 * Register a new user
 */
export async function POST(request: NextRequest) {
  try {
    const { email, password, full_name } = await request.json();

    // Validate input
    const errors: ValidationError[] = [];

    if (!email || !validateEmail(email)) {
      errors.push({ field: 'email', message: 'Valid email is required' });
    }

    const passwordValidation = validatePassword(password);
    if (!passwordValidation.isValid) {
      errors.push({ field: 'password', message: passwordValidation.errors.join(', ') });
    }

    if (full_name && typeof full_name !== 'string') {
      errors.push({ field: 'full_name', message: 'Full name must be a string' });
    }

    if (errors.length > 0) {
      return validationErrorResponse(errors);
    }

    // Check if user already exists
    const { data: existingUsersData } = await supabaseAdmin.auth.admin.listUsers();
    const existingUser = existingUsersData.users.find(u => u.email === email);
    if (existingUser) {
      return errorResponse('User already exists', 409);
    }

    // Create user with Supabase Auth
    const { data: authData, error: authError } = await supabaseAdmin.auth.admin.createUser({
      email,
      password,
      user_metadata: {
        full_name: full_name || null,
      },
      email_confirm: true, // Skip email confirmation for development
    });

    if (authError) {
      console.error('Auth error:', authError);
      return errorResponse('Failed to create user', 500);
    }

    if (!authData.user) {
      return errorResponse('User creation failed', 500);
    }

    // Generate JWT token
    const token = generateJWT({
      id: authData.user.id,
      email: authData.user.email!,
    });

    const expiresIn = getTokenExpirationTime();

    // Return user data with token
    return successResponse({
      user: {
        id: authData.user.id,
        email: authData.user.email!,
        full_name: authData.user.user_metadata?.full_name || null,
        created_at: authData.user.created_at,
        updated_at: authData.user.updated_at,
      },
      token,
      expires_in: expiresIn,
    }, 'User created successfully', 201);

  } catch (error) {
    console.error('Signup error:', error);
    return errorResponse('Internal server error', 500);
  }
}
