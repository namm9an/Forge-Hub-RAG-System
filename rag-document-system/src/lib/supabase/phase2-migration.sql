-- Phase 2 Database Updates

-- Add new fields to documents table
ALTER TABLE public.documents 
ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS processing_error TEXT,
ADD COLUMN IF NOT EXISTS file_metadata JSONB DEFAULT '{}'::jsonb;

-- Create processing_jobs table
CREATE TABLE IF NOT EXISTS public.processing_jobs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    document_id UUID REFERENCES public.documents(id) ON DELETE CASCADE NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create index for processing_jobs
CREATE INDEX IF NOT EXISTS processing_jobs_status_idx ON public.processing_jobs(status);
CREATE INDEX IF NOT EXISTS processing_jobs_document_id_idx ON public.processing_jobs(document_id);
CREATE INDEX IF NOT EXISTS processing_jobs_created_at_idx ON public.processing_jobs(created_at);

-- Enable RLS for processing_jobs
ALTER TABLE public.processing_jobs ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for processing_jobs
CREATE POLICY "Users can view own processing jobs only" ON public.processing_jobs
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.documents
            WHERE documents.id = processing_jobs.document_id
            AND documents.user_id = auth.uid()
        )
    );

-- Only system can insert/update/delete processing jobs
CREATE POLICY "System can manage processing jobs" ON public.processing_jobs
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Create trigger for processing_jobs updated_at
CREATE TRIGGER handle_updated_at_processing_jobs
    BEFORE UPDATE ON public.processing_jobs
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- Create Supabase Storage bucket for documents
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'documents', 
    'documents', 
    false, 
    52428800, -- 50MB
    ARRAY[
        'application/pdf',
        'text/plain',
        'text/markdown', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]::text[]
)
ON CONFLICT (id) DO UPDATE SET
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

-- Create storage policies for documents bucket
CREATE POLICY "Users can upload their own documents" ON storage.objects
    FOR INSERT WITH CHECK (
        bucket_id = 'documents' 
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "Users can view their own documents" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'documents' 
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "Users can delete their own documents" ON storage.objects
    FOR DELETE USING (
        bucket_id = 'documents' 
        AND auth.uid()::text = (storage.foldername(name))[1]
    );
