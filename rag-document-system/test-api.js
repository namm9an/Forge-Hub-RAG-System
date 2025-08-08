import fetch from 'node-fetch';

const BASE_URL = 'http://localhost:3000';

async function testAPI() {
  console.log('🚀 Testing Forge-Hub RAG System API endpoints...\n');

  // Test 1: Health Check
  console.log('1. Testing Health Check...');
  try {
    const response = await fetch(`${BASE_URL}/api/health`);
    const data = await response.json();
    console.log('✅ Health Check:', response.status, data.data?.status || data.error);
  } catch (error) {
    console.log('❌ Health Check failed:', error.message);
  }

  // Test 2: System Info
  console.log('\n2. Testing System Info...');
  try {
    const response = await fetch(`${BASE_URL}/api/system/info`);
    const data = await response.json();
    console.log('✅ System Info:', response.status, data.data?.version || data.error);
  } catch (error) {
    console.log('❌ System Info failed:', error.message);
  }

  // Test 3: User Signup
  console.log('\n3. Testing User Signup...');
  const testUser = {
    email: 'test@example.com',
    password: 'SecurePass123!',
    full_name: 'Test User'
  };

  try {
    const response = await fetch(`${BASE_URL}/api/auth/signup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(testUser)
    });
    const data = await response.json();
    console.log('✅ User Signup:', response.status, data.success ? 'Success' : data.error);
    
    if (data.success && data.data?.token) {
      global.authToken = data.data.token;
      console.log('🔑 Auth token saved for subsequent tests');
    }
  } catch (error) {
    console.log('❌ User Signup failed:', error.message);
  }

  // Test 4: User Signin
  console.log('\n4. Testing User Signin...');
  try {
    const response = await fetch(`${BASE_URL}/api/auth/signin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        email: testUser.email,
        password: testUser.password
      })
    });
    const data = await response.json();
    console.log('✅ User Signin:', response.status, data.success ? 'Success' : data.error);
    
    if (data.success && data.data?.token) {
      global.authToken = data.data.token;
    }
  } catch (error) {
    console.log('❌ User Signin failed:', error.message);
  }

  // Test 5: Get User Profile
  if (global.authToken) {
    console.log('\n5. Testing Get User Profile...');
    try {
      const response = await fetch(`${BASE_URL}/api/auth/me`, {
        headers: {
          'Authorization': `Bearer ${global.authToken}`
        }
      });
      const data = await response.json();
      console.log('✅ Get User Profile:', response.status, data.success ? 'Success' : data.error);
    } catch (error) {
      console.log('❌ Get User Profile failed:', error.message);
    }

    // Test 6: List Documents
    console.log('\n6. Testing List Documents...');
    try {
      const response = await fetch(`${BASE_URL}/api/documents`, {
        headers: {
          'Authorization': `Bearer ${global.authToken}`
        }
      });
      const data = await response.json();
      console.log('✅ List Documents:', response.status, data.success ? 'Success' : data.error);
    } catch (error) {
      console.log('❌ List Documents failed:', error.message);
    }

    // Test 7: Search Documents
    console.log('\n7. Testing Search Documents...');
    try {
      const response = await fetch(`${BASE_URL}/api/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${global.authToken}`
        },
        body: JSON.stringify({
          query: 'test search query'
        })
      });
      const data = await response.json();
      console.log('✅ Search Documents:', response.status, data.success ? 'Success' : data.error);
    } catch (error) {
      console.log('❌ Search Documents failed:', error.message);
    }
  } else {
    console.log('\n⚠️  Skipping authenticated tests (no auth token)');
  }

  console.log('\n🎉 API testing completed!');
}

// Run the tests
testAPI().catch(console.error);
