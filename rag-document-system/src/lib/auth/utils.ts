import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { User } from '@/lib/types';

const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secure-32-character-secret-key-here-2024';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '24h';

if (!process.env.JWT_SECRET) {
  console.warn('JWT_SECRET environment variable is not set, using default value');
}

export interface JWTPayload {
  userId: string;
  email: string;
  iat?: number;
  exp?: number;
}

/**
 * Generate a JWT token for the given user
 * @param user - User object containing id and email
 * @returns JWT token string
 */
export function generateJWT(user: { id: string; email: string }): string {
  try {
    const payload: JWTPayload = {
      userId: user.id,
      email: user.email,
    };

    const options: jwt.SignOptions = {
      expiresIn: JWT_EXPIRES_IN,
      issuer: 'rag-document-system',
      audience: 'rag-document-system-users',
    };

    return jwt.sign(payload, JWT_SECRET, options);
  } catch (error) {
    console.error('Error generating JWT:', error);
    throw new Error('Failed to generate authentication token');
  }
}

/**
 * Verify and decode a JWT token
 * @param token - JWT token string
 * @returns Decoded JWT payload
 */
export function verifyJWT(token: string): JWTPayload {
  try {
    const decoded = jwt.verify(token, JWT_SECRET, {
      issuer: 'rag-document-system',
      audience: 'rag-document-system-users',
    }) as JWTPayload;

    return decoded;
  } catch (error) {
    console.error('Error verifying JWT:', error);
    
    if (error instanceof jwt.TokenExpiredError) {
      throw new Error('Authentication token has expired');
    } else if (error instanceof jwt.JsonWebTokenError) {
      throw new Error('Invalid authentication token');
    } else {
      throw new Error('Authentication failed');
    }
  }
}

/**
 * Hash a password using bcrypt
 * @param password - Plain text password
 * @returns Promise<string> - Hashed password
 */
export async function hashPassword(password: string): Promise<string> {
  try {
    if (!password || password.trim().length === 0) {
      throw new Error('Password cannot be empty');
    }

    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(password, saltRounds);
    return hashedPassword;
  } catch (error) {
    console.error('Error hashing password:', error);
    throw new Error('Failed to hash password');
  }
}

/**
 * Verify a password against its hash
 * @param password - Plain text password
 * @param hash - Hashed password
 * @returns Promise<boolean> - True if password matches
 */
export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  try {
    if (!password || !hash) {
      return false;
    }

    const isMatch = await bcrypt.compare(password, hash);
    return isMatch;
  } catch (error) {
    console.error('Error verifying password:', error);
    return false;
  }
}

/**
 * Extract JWT token from Authorization header
 * @param authHeader - Authorization header value
 * @returns JWT token string or null
 */
export function extractTokenFromHeader(authHeader: string | null): string | null {
  if (!authHeader) {
    return null;
  }

  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0].toLowerCase() !== 'bearer') {
    return null;
  }

  return parts[1];
}

/**
 * Validate password strength
 * @param password - Password to validate
 * @returns Object with validation result and errors
 */
export function validatePassword(password: string): { isValid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!password) {
    errors.push('Password is required');
    return { isValid: false, errors };
  }

  if (password.length < 8) {
    errors.push('Password must be at least 8 characters long');
  }

  if (password.length > 128) {
    errors.push('Password must not exceed 128 characters');
  }

  if (!/[A-Z]/.test(password)) {
    errors.push('Password must contain at least one uppercase letter');
  }

  if (!/[a-z]/.test(password)) {
    errors.push('Password must contain at least one lowercase letter');
  }

  if (!/[0-9]/.test(password)) {
    errors.push('Password must contain at least one number');
  }

  if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
    errors.push('Password must contain at least one special character');
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
}

/**
 * Validate email format
 * @param email - Email to validate
 * @returns boolean - True if email is valid
 */
export function validateEmail(email: string): boolean {
  if (!email || typeof email !== 'string') {
    return false;
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email.trim().toLowerCase());
}

/**
 * Get token expiration time in milliseconds
 * @returns number - Token expiration time in milliseconds
 */
export function getTokenExpirationTime(): number {
  // Convert JWT_EXPIRES_IN to milliseconds
  const expiresIn = JWT_EXPIRES_IN;
  
  if (expiresIn.endsWith('h')) {
    return parseInt(expiresIn) * 60 * 60 * 1000;
  } else if (expiresIn.endsWith('m')) {
    return parseInt(expiresIn) * 60 * 1000;
  } else if (expiresIn.endsWith('s')) {
    return parseInt(expiresIn) * 1000;
  } else if (expiresIn.endsWith('d')) {
    return parseInt(expiresIn) * 24 * 60 * 60 * 1000;
  }
  
  // Default to 24 hours if format is not recognized
  return 24 * 60 * 60 * 1000;
}
