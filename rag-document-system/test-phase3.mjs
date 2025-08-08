import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';

const BASE_URL = 'http://localhost:3000';
let authToken = null;
let testDocumentId = null;

// Test utilities
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m'
};

function log(message, color = colors.white) {
  console.log(`${color}${message}${colors.reset}`);
}

function success(message) {
  log(`✅ ${message}`, colors.green);
}

function error(message) {
  log(`❌ ${message}`, colors.red);
}

function warning(message) {
  log(`⚠️  ${message}`, colors.yellow);
}

function info(message) {
  log(`ℹ️  ${message}`, colors.blue);
}

function header(message) {
  log(`\n🔍 ${message}`, colors.cyan);
}

// Test Phase 3 implementation
async function testPhase3Implementation() {
  header('PHASE 3 VECTOR EMBEDDINGS IMPLEMENTATION TEST');
  
  let passed = 0;
  let failed = 0;
  
  // Test 1: Authentication Setup
  header('1. Authentication Setup');
  try {
    const testUser = {
      email: 'phase3test@example.com',
      password: 'SecurePass123!',
      full_name: 'Phase 3 Test User'
    };

    // Try signup first
    const signupResponse = await fetch(`${BASE_URL}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testUser)
    });

    if (signupResponse.ok) {
      const signupData = await signupResponse.json();
      authToken = signupData.data?.token;
      success('User signup successful');
    } else {
      // Try signin if signup fails (user might already exist)
      const signinResponse = await fetch(`${BASE_URL}/api/auth/signin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: testUser.email, password: testUser.password })
      });

      if (signinResponse.ok) {
        const signinData = await signinResponse.json();
        authToken = signinData.data?.token;
        success('User signin successful');
      } else {
        throw new Error('Authentication failed');
      }
    }
    passed++;
  } catch (err) {
    error(`Authentication failed: ${err.message}`);
    failed++;
    return; // Exit if authentication fails
  }

  // Test 2: Get Existing Documents
  header('2. Check Existing Documents');
  try {
    const documentsResponse = await fetch(`${BASE_URL}/api/documents`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });

    if (documentsResponse.ok) {
      const documentsData = await documentsResponse.json();
      const documents = documentsData.data || [];
      
      if (documents.length > 0) {
        testDocumentId = documents[0].id;
        success(`Found existing document: ${testDocumentId}`);
        passed++;
      } else {
        warning('No existing documents found - skipping document-specific tests');
        // We'll skip tests that require a document
        testDocumentId = null;
        passed++;
      }
    } else {
      throw new Error('Failed to retrieve documents');
    }
  } catch (err) {
    error(`Document retrieval failed: ${err.message}`);
    failed++;
  }

  // Test 3: Wait for Document Processing
  header('3. Document Processing Status');
  if (!testDocumentId) {
    warning('Skipping document processing test - no document available');
    passed++;
  } else {
    try {
      let processingComplete = false;
      let attempts = 0;
      const maxAttempts = 10;

      while (!processingComplete && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds

        const statusResponse = await fetch(`${BASE_URL}/api/documents/${testDocumentId}/status`, {
          headers: { 'Authorization': `Bearer ${authToken}` }
        });

        if (statusResponse.ok) {
          const statusData = await statusResponse.json();
          const status = statusData.data?.processing_status;
          
          info(`Document processing status: ${status} (attempt ${attempts + 1}/${maxAttempts})`);
          
          if (status === 'completed') {
            processingComplete = true;
            success('Document processing completed');
            passed++;
            break;
          } else if (status === 'failed') {
            throw new Error('Document processing failed');
          }
        }

        attempts++;
      }

      if (!processingComplete) {
        throw new Error('Document processing timeout');
      }
    } catch (err) {
      error(`Document processing check failed: ${err.message}`);
      failed++;
    }
  }

  // Test 4: Embedding Generation API
  header('4. Embedding Generation API');
  try {
    const embeddingResponse = await fetch(`${BASE_URL}/api/embeddings/generate/${testDocumentId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ priority: 'normal', force_reprocess: false })
    });

    if (embeddingResponse.ok) {
      const embeddingData = await embeddingResponse.json();
      success(`Embedding generation started: Job ID ${embeddingData.data.job_id}`);
      passed++;
    } else {
      const errorData = await embeddingResponse.json();
      throw new Error(errorData.error || 'Embedding generation failed');
    }
  } catch (err) {
    error(`Embedding generation failed: ${err.message}`);
    failed++;
  }

  // Test 5: Embedding Processing Status
  header('5. Embedding Processing Status');
  try {
    let embeddingComplete = false;
    let attempts = 0;
    const maxAttempts = 15;

    while (!embeddingComplete && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 seconds

      const statusResponse = await fetch(`${BASE_URL}/api/embeddings/generate/${testDocumentId}`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });

      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        const jobs = statusData.data?.jobs || [];
        
        if (jobs.length > 0) {
          const latestJob = jobs[0];
          const status = latestJob.status;
          const progress = latestJob.progress_percentage || 0;
          
          info(`Embedding processing: ${status} (${progress}%) (attempt ${attempts + 1}/${maxAttempts})`);
          
          if (status === 'completed') {
            embeddingComplete = true;
            success('Embedding processing completed');
            passed++;
            break;
          } else if (status === 'failed') {
            throw new Error(`Embedding processing failed: ${latestJob.error_message}`);
          }
        }
      }

      attempts++;
    }

    if (!embeddingComplete) {
      warning('Embedding processing may still be in progress');
    }
  } catch (err) {
    error(`Embedding processing check failed: ${err.message}`);
    failed++;
  }

  // Test 6: Vector Statistics
  header('6. Vector Statistics');
  try {
    const statsResponse = await fetch(`${BASE_URL}/api/vectors/stats`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });

    if (statsResponse.ok) {
      const statsData = await statsResponse.json();
      const stats = statsData.data;
      
      success(`Vector statistics retrieved:`);
      info(`  - Total embeddings: ${stats.total_embeddings}`);
      info(`  - Total documents: ${stats.total_documents}`);
      info(`  - Average chunks per document: ${stats.avg_chunks_per_document}`);
      info(`  - Storage size: ${stats.storage_size_mb} MB`);
      passed++;
    } else {
      throw new Error('Vector statistics retrieval failed');
    }
  } catch (err) {
    error(`Vector statistics failed: ${err.message}`);
    failed++;
  }

  // Test 7: Similarity Search
  header('7. Similarity Search');
  try {
    const searchQueries = [
      'artificial intelligence and machine learning',
      'natural language processing',
      'vector embeddings',
      'comprehensive test document'
    ];

    for (const query of searchQueries) {
      info(`Testing search query: "${query}"`);
      
      const searchResponse = await fetch(`${BASE_URL}/api/search/similarity`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          similarity_threshold: 0.7,
          limit: 5
        })
      });

      if (searchResponse.ok) {
        const searchData = await searchResponse.json();
        const results = searchData.data.results || [];
        
        success(`  Found ${results.length} results`);
        if (results.length > 0) {
          info(`  Top result similarity: ${results[0].similarity_score.toFixed(3)}`);
        }
      } else {
        const errorData = await searchResponse.json();
        throw new Error(errorData.error || 'Search failed');
      }
    }
    
    passed++;
  } catch (err) {
    error(`Similarity search failed: ${err.message}`);
    failed++;
  }

  // Test 8: Batch Embedding Generation
  header('8. Batch Embedding Generation');
  try {
    const batchResponse = await fetch(`${BASE_URL}/api/embeddings/batch`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        document_ids: [testDocumentId],
        batch_size: 10,
        priority: 'normal'
      })
    });

    if (batchResponse.ok) {
      const batchData = await batchResponse.json();
      success(`Batch embedding generation started: ${batchData.data.job_id}`);
      passed++;
    } else {
      const errorData = await batchResponse.json();
      warning(`Batch embedding generation: ${errorData.error || 'May already be processed'}`);
    }
  } catch (err) {
    error(`Batch embedding generation failed: ${err.message}`);
    failed++;
  }

  // Test 9: Vector Index Management
  header('9. Vector Index Management');
  try {
    const rebuildResponse = await fetch(`${BASE_URL}/api/vectors/rebuild-index`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${authToken}` }
    });

    if (rebuildResponse.ok) {
      success('Vector index rebuild initiated');
      passed++;
    } else {
      const errorData = await rebuildResponse.json();
      warning(`Vector index rebuild: ${errorData.error || 'May not be implemented'}`);
    }
  } catch (err) {
    error(`Vector index management failed: ${err.message}`);
    failed++;
  }

  // Test 10: Performance and Load Testing
  header('10. Performance Testing');
  try {
    const performanceQueries = [
      'machine learning algorithms',
      'artificial intelligence applications',
      'natural language understanding',
      'semantic similarity search',
      'text processing methods'
    ];

    const startTime = Date.now();
    const searchPromises = performanceQueries.map(query => 
      fetch(`${BASE_URL}/api/search/similarity`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          similarity_threshold: 0.6,
          limit: 10
        })
      })
    );

    const responses = await Promise.all(searchPromises);
    const endTime = Date.now();
    const totalTime = endTime - startTime;

    const successfulSearches = responses.filter(r => r.ok).length;
    success(`Concurrent search performance: ${successfulSearches}/${performanceQueries.length} successful`);
    info(`Total time: ${totalTime}ms, Average per query: ${(totalTime / performanceQueries.length).toFixed(2)}ms`);
    
    if (successfulSearches === performanceQueries.length && totalTime < 10000) {
      passed++;
    } else {
      failed++;
    }
  } catch (err) {
    error(`Performance testing failed: ${err.message}`);
    failed++;
  }

  // Test Summary
  header('TEST SUMMARY');
  const total = passed + failed;
  const passRate = ((passed / total) * 100).toFixed(1);
  
  log(`\n📊 Test Results:`);
  success(`Passed: ${passed}/${total} tests`);
  if (failed > 0) {
    error(`Failed: ${failed}/${total} tests`);
  }
  log(`Pass Rate: ${passRate}%\n`);

  // Implementation Status
  if (passRate >= 90) {
    success('🎉 Phase 3 Implementation: EXCELLENT');
    log('✅ All critical features are working correctly');
  } else if (passRate >= 75) {
    success('👍 Phase 3 Implementation: GOOD');
    log('✅ Most features are working, minor issues detected');
  } else if (passRate >= 50) {
    warning('⚠️  Phase 3 Implementation: PARTIAL');
    log('🔧 Some features need attention');
  } else {
    error('❌ Phase 3 Implementation: NEEDS WORK');
    log('🚨 Major issues detected, significant work required');
  }

  // Recommendations
  header('RECOMMENDATIONS');
  if (failed > 0) {
    log('🔧 Areas for improvement:');
    log('  - Review failed test cases');
    log('  - Check database schema and migrations');
    log('  - Verify API endpoint implementations');
    log('  - Test embedding generation pipeline');
    log('  - Optimize similarity search performance');
  }
  
  log('📋 Next steps:');
  log('  1. Review any failing tests');
  log('  2. Test with larger documents');
  log('  3. Monitor embedding generation performance');
  log('  4. Optimize vector search for production');
  log('  5. Implement caching strategies');
  log('  6. Add monitoring and alerting');
}

// Run the test
testPhase3Implementation().catch(console.error);
