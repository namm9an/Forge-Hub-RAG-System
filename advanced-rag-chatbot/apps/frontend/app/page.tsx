export default function HomePage() {
  return (
    <main className="container">
      <h1>Advanced RAG Chatbot</h1>
      <p>Frontend is up. Connect to backend at {process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'}.</p>
    </main>
  );
}

