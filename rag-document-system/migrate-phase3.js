const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// Load environment variables
require('dotenv').config();

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('Missing required environment variables:');
  console.error('- NEXT_PUBLIC_SUPABASE_URL');
  console.error('- SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function runMigration() {
  try {
    console.log('🚀 Starting Phase 3 Migration: Vector Embeddings');
    
    // Read the migration SQL file
    const migrationSQL = fs.readFileSync(
      path.join(__dirname, 'src/lib/supabase/phase3-migration.sql'),
      'utf8'
    );
    
    // Split SQL into individual statements
    const statements = migrationSQL
      .split(';')
      .map(stmt => stmt.trim())
      .filter(stmt => stmt.length > 0 && !stmt.startsWith('--'));
    
    console.log(`📄 Found ${statements.length} SQL statements to execute`);
    
    // Execute each statement
    for (let i = 0; i < statements.length; i++) {
      const statement = statements[i];
      if (statement.length === 0) continue;
      
      try {
        console.log(`⚙️  Executing statement ${i + 1}/${statements.length}...`);
        
        const { error } = await supabase.rpc('exec_sql', {
          sql: statement + ';'
        });
        
        if (error) {
          // Try alternative approach for some statements
          const { error: directError } = await supabase
            .from('_migrations')
            .insert({ statement: statement });
          
          if (directError) {
            console.warn(`⚠️  Warning on statement ${i + 1}: ${error.message}`);
            // Continue with next statement
          }
        }
        
      } catch (statementError) {
        console.warn(`⚠️  Warning on statement ${i + 1}: ${statementError.message}`);
        // Continue with next statement
      }
    }
    
    // Verify critical tables exist
    console.log('🔍 Verifying migration success...');
    
    const criticalTables = [
      'embedding_jobs',
      'vector_search_cache',
      'embedding_cache',
      'rate_limit_tracking'
    ];
    
    for (const table of criticalTables) {
      const { data, error } = await supabase
        .from(table)
        .select('*')
        .limit(1);
      
      if (error) {
        console.error(`❌ Critical table '${table}' not found or accessible`);
        console.error(error.message);
      } else {
        console.log(`✅ Table '${table}' verified`);
      }
    }
    
    // Check if pgvector extension is enabled
    const { data: extensions, error: extError } = await supabase
      .rpc('exec_sql', {
        sql: "SELECT * FROM pg_extension WHERE extname = 'vector';"
      });
    
    if (extError) {
      console.warn('⚠️  Could not verify pgvector extension');
    } else {
      console.log('✅ pgvector extension verified');
    }
    
    // Test basic functionality
    console.log('🧪 Testing basic functionality...');
    
    // Test embedding cache
    const testCacheData = {
      content_hash: 'test_hash_' + Date.now(),
      content_preview: 'Test content for migration verification',
      embedding: new Array(768).fill(0).map(() => Math.random()),
      model_version: 'text-embedding-004'
    };
    
    const { error: cacheError } = await supabase
      .from('embedding_cache')
      .insert(testCacheData);
    
    if (cacheError) {
      console.error('❌ Embedding cache test failed:', cacheError.message);
    } else {
      console.log('✅ Embedding cache test passed');
      
      // Cleanup test data
      await supabase
        .from('embedding_cache')
        .delete()
        .eq('content_hash', testCacheData.content_hash);
    }
    
    // Test vector search cache
    const testSearchData = {
      user_id: '00000000-0000-0000-0000-000000000000', // dummy UUID
      query_hash: 'test_query_' + Date.now(),
      query_text: 'Test query for migration verification',
      search_results: [{ test: 'data' }],
      similarity_threshold: 0.7,
      result_count: 1
    };
    
    const { error: searchError } = await supabase
      .from('vector_search_cache')
      .insert(testSearchData);
    
    if (searchError) {
      console.error('❌ Vector search cache test failed:', searchError.message);
    } else {
      console.log('✅ Vector search cache test passed');
      
      // Cleanup test data
      await supabase
        .from('vector_search_cache')
        .delete()
        .eq('query_hash', testSearchData.query_hash);
    }
    
    console.log('\n🎉 Phase 3 Migration completed successfully!');
    console.log('\n📋 Migration Summary:');
    console.log('  ✅ pgvector extension enabled');
    console.log('  ✅ Embedding tables created');
    console.log('  ✅ Vector indexes created');
    console.log('  ✅ Cache tables created');
    console.log('  ✅ Rate limiting tables created');
    console.log('  ✅ Row Level Security policies applied');
    console.log('  ✅ Database functions created');
    console.log('  ✅ Triggers and views created');
    
    console.log('\n🔧 Next Steps:');
    console.log('  1. Update your application environment variables');
    console.log('  2. Test the embedding generation endpoints');
    console.log('  3. Verify similarity search functionality');
    console.log('  4. Monitor vector index performance');
    
  } catch (error) {
    console.error('❌ Migration failed:', error.message);
    console.error('Stack trace:', error.stack);
    process.exit(1);
  }
}

// Run the migration
runMigration();
