import { NextResponse } from 'next/server';
import { ApiResponse, ValidationError } from '@/lib/types';

/**
 * Create a successful API response
 * @param data - Response data
 * @param message - Optional success message
 * @param status - HTTP status code (default: 200)
 * @returns NextResponse with success data
 */
export function successResponse<T>(
  data: T,
  message?: string,
  status: number = 200
): NextResponse<ApiResponse<T>> {
  return NextResponse.json(
    {
      success: true,
      data,
      message,
    },
    { status }
  );
}

/**
 * Create an error API response
 * @param error - Error message
 * @param status - HTTP status code (default: 400)
 * @returns NextResponse with error data
 */
export function errorResponse(
  error: string,
  status: number = 400
): NextResponse<ApiResponse> {
  return NextResponse.json(
    {
      success: false,
      error,
    },
    { status }
  );
}

/**
 * Create a validation error response
 * @param errors - Array of validation errors
 * @returns NextResponse with validation errors
 */
export function validationErrorResponse(
  errors: ValidationError[]
): NextResponse<ApiResponse> {
  return NextResponse.json(
    {
      success: false,
      error: 'Validation failed',
      data: { errors },
    },
    { status: 422 }
  );
}

/**
 * Create an unauthorized response
 * @param message - Optional error message
 * @returns NextResponse with 401 status
 */
export function unauthorizedResponse(
  message: string = 'Unauthorized'
): NextResponse<ApiResponse> {
  return NextResponse.json(
    {
      success: false,
      error: message,
    },
    { status: 401 }
  );
}

/**
 * Create a forbidden response
 * @param message - Optional error message
 * @returns NextResponse with 403 status
 */
export function forbiddenResponse(
  message: string = 'Forbidden'
): NextResponse<ApiResponse> {
  return NextResponse.json(
    {
      success: false,
      error: message,
    },
    { status: 403 }
  );
}

/**
 * Create a not found response
 * @param message - Optional error message
 * @returns NextResponse with 404 status
 */
export function notFoundResponse(
  message: string = 'Not found'
): NextResponse<ApiResponse> {
  return NextResponse.json(
    {
      success: false,
      error: message,
    },
    { status: 404 }
  );
}

/**
 * Create a rate limit response
 * @param message - Optional error message
 * @returns NextResponse with 429 status
 */
export function rateLimitResponse(
  message: string = 'Rate limit exceeded'
): NextResponse<ApiResponse> {
  return NextResponse.json(
    {
      success: false,
      error: message,
    },
    { status: 429 }
  );
}

/**
 * Create an internal server error response
 * @param message - Optional error message
 * @returns NextResponse with 500 status
 */
export function internalServerErrorResponse(
  message: string = 'Internal server error'
): NextResponse<ApiResponse> {
  return NextResponse.json(
    {
      success: false,
      error: message,
    },
    { status: 500 }
  );
}

/**
 * Handle and format API errors
 * @param error - Error object
 * @returns NextResponse with appropriate error response
 */
export function handleApiError(error: unknown): NextResponse<ApiResponse> {
  console.error('API Error:', error);

  if (error instanceof Error) {
    // Handle specific error types
    if (error.message.includes('Authentication')) {
      return unauthorizedResponse(error.message);
    }
    
    if (error.message.includes('Forbidden')) {
      return forbiddenResponse(error.message);
    }
    
    if (error.message.includes('Not found')) {
      return notFoundResponse(error.message);
    }
    
    if (error.message.includes('Rate limit')) {
      return rateLimitResponse(error.message);
    }
    
    if (error.message.includes('Validation')) {
      return errorResponse(error.message, 422);
    }
    
    return errorResponse(error.message);
  }

  return internalServerErrorResponse();
}

/**
 * Create a paginated response
 * @param data - Array of data items
 * @param pagination - Pagination information
 * @returns NextResponse with paginated data
 */
export function paginatedResponse<T>(
  data: T[],
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  }
): NextResponse<ApiResponse> {
  return NextResponse.json(
    {
      success: true,
      data,
      pagination,
    },
    { status: 200 }
  );
}
