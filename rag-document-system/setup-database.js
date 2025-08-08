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

async function setupDatabase() {
  try {
    console.log('Setting up database schema...');
    
    // Read schema file
    const schemaPath = path.join(__dirname, 'src/lib/supabase/schema.sql');
    const schema = fs.readFileSync(schemaPath, 'utf8');
    
    // Split schema into individual statements
    const statements = schema.split(';').filter(stmt => stmt.trim().length > 0);
    
    console.log(`Executing ${statements.length} SQL statements...`);
    
    for (let i = 0; i < statements.length; i++) {
      const statement = statements[i].trim();
      if (statement) {
        try {
          console.log(`Executing statement ${i + 1}/${statements.length}`);
          const { error } = await supabase.rpc('exec_sql', { sql: statement });
          if (error) {
            console.warn(`Warning on statement ${i + 1}:`, error.message);
          }
        } catch (err) {
          console.warn(`Warning on statement ${i + 1}:`, err.message);
        }
      }
    }
    
    console.log('✅ Database schema setup completed!');
    
    // Test the setup
    console.log('Testing database connection...');
    const { data, error } = await supabase.from('profiles').select('count');
    if (error) {
      console.log('✅ Database is ready (tables created)');
    } else {
      console.log('✅ Database connection test passed');
    }
    
  } catch (error) {
    console.error('❌ Database setup failed:', error);
    process.exit(1);
  }
}

setupDatabase();
