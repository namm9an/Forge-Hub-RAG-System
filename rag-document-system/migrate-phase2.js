const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// Load environment variables
require('dotenv').config({ path: '.env.local' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('Missing Supabase environment variables');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function runMigration() {
  try {
    console.log('Running Phase 2 database migration...');
    
    // Add new fields to documents table
    console.log('Adding new fields to documents table...');
    try {
      const { error } = await supabase.rpc('exec_sql', { 
        sql: `ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0, ADD COLUMN IF NOT EXISTS processing_error TEXT, ADD COLUMN IF NOT EXISTS file_metadata JSONB DEFAULT '{}'::jsonb;` 
      });
      if (error) console.warn('Warning:', error.message);
    } catch (err) {
      console.warn('Warning:', err.message);
    }
    
    // Manually execute the key SQL statements
    console.log('Creating processing_jobs table...');
    const createProcessingJobsTable = `
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
    `;
    
    const { error: createTableError } = await supabase.rpc('exec_sql', { sql: createProcessingJobsTable });
    if (createTableError) console.warn('Warning:', createTableError.message);
    
    console.log('✅ Phase 2 database migration completed!');
    
  } catch (error) {
    console.error('❌ Database migration failed:', error);
    console.log('Note: You may need to run the migration manually in the Supabase dashboard.');
    process.exit(1);
  }
}

runMigration();
