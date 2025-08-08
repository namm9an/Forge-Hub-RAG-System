export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-8">
            Forge-Hub RAG System
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            Phase 1 Foundation - Next.js + Supabase + Gemini 2.5
          </p>
        </div>

        <div className="bg-white shadow-md rounded-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">System Status</h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Database</span>
              <span className="text-green-600">✓ Connected</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Authentication</span>
              <span className="text-green-600">✓ Ready</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">API Endpoints</span>
              <span className="text-green-600">✓ Available</span>
            </div>
          </div>
        </div>

        <div className="bg-white shadow-md rounded-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Available APIs</h2>
          <div className="space-y-2">
            <div className="flex items-center">
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-sm mr-3">GET</span>
              <code className="text-sm bg-gray-100 px-2 py-1 rounded">/api/health</code>
            </div>
            <div className="flex items-center">
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm mr-3">POST</span>
              <code className="text-sm bg-gray-100 px-2 py-1 rounded">/api/auth/signup</code>
            </div>
            <div className="flex items-center">
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm mr-3">POST</span>
              <code className="text-sm bg-gray-100 px-2 py-1 rounded">/api/auth/signin</code>
            </div>
            <div className="flex items-center">
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-sm mr-3">GET</span>
              <code className="text-sm bg-gray-100 px-2 py-1 rounded">/api/auth/me</code>
            </div>
            <div className="flex items-center">
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-sm mr-3">GET</span>
              <code className="text-sm bg-gray-100 px-2 py-1 rounded">/api/documents</code>
            </div>
            <div className="flex items-center">
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm mr-3">POST</span>
              <code className="text-sm bg-gray-100 px-2 py-1 rounded">/api/documents/upload</code>
            </div>
          </div>
        </div>

        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Next Steps</h2>
          <ol className="list-decimal list-inside space-y-2 text-gray-600">
            <li>Set up database tables using the schema.sql file</li>
            <li>Add your Gemini API key to the environment variables</li>
            <li>Test the API endpoints</li>
            <li>Ready for Phase 2: Document Processing</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
