import { NextRequest } from 'next/server';
import { successResponse, unauthorizedResponse, errorResponse, validationErrorResponse } from '@/lib/utils/response';
import { supabaseAdmin } from '@/lib/supabase/server';
import { verifyJWT, extractTokenFromHeader } from '@/lib/auth/utils';
import { ValidationError } from '@/lib/types';
import { uploadFile } from '@/lib/storage/supabaseStorage';
import { createDocument } from '@/lib/database/documentQueries';
import { addProcessingJob } from '@/lib/processing/jobQueue';
import { validateFile } from '@/lib/storage/fileValidator';

/**
 * POST /api/documents/upload
 * Upload document endpoint
 */
export async function POST(request: NextRequest) {
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

    // Get form data
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const title = formData.get('title') as string;

    // Validate file
    const validation = validateFile(file);
    if (!validation.valid) {
      return validationErrorResponse([{ field: 'file', message: validation.error || 'Invalid file' }]);
    }

    // Upload file to storage
    const uploadResult = await uploadFile(file, decoded.userId, file.name);

    // Create document record
    const documentData = {
      user_id: decoded.userId,
      title: title.trim(),
      file_name: file.name,
      file_type: file.type,
      file_size: file.size,
      upload_path: uploadResult.path,
    };
    const document = await createDocument(documentData);

    // Add processing job
    await addProcessingJob(document.id);

    // Return success response
    return successResponse({
      document: {
        id: document.id,
        title: document.title,
        file_name: document.file_name,
        file_type: document.file_type,
        file_size: document.file_size,
        processing_status: document.processing_status,
        created_at: document.created_at,
      },
      message: 'Document uploaded successfully. Processing will begin shortly.',
    }, 'Document uploaded successfully', 201);

  } catch (error) {
    console.error('Upload error:', error);
    
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
