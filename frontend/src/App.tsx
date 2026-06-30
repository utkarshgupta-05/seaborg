import { useState } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { SearchBar } from './components/SearchBar';
import { AnswerPanel } from './components/AnswerPanel';
import { ChartPanel } from './components/ChartPanel';
import { useChat } from './hooks/useChat';

function App() {
  const { response, messages, isLoading, error, sendMessage, clearResponse, isBackendConnected } = useChat();
  const [newQueryAdded, setNewQueryAdded] = useState<string | null>(null);

  const handleSend = async (message: string) => {
    setNewQueryAdded(message);
    await sendMessage(message);
    // Reset after a brief moment to allow effect to trigger
    setTimeout(() => setNewQueryAdded(null), 100);
  };

  const handleNewQuery = () => {
    clearResponse();
  };

  return (
    <div className="layout-container">
      <Header isBackendConnected={isBackendConnected} />
      
      <Sidebar 
        onNewQuery={handleNewQuery} 
        onRerunQuery={handleSend}
        newQueryAdded={newQueryAdded}
      />
      
      <main className="main-area">
        <div style={{ maxWidth: '100%', margin: '0 auto', width: '100%', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <SearchBar onSend={handleSend} isLoading={isLoading} />
          
          {error && (
            <div style={{ 
              backgroundColor: '#fee2e2', 
              color: '#ef4444', 
              padding: '12px 16px', 
              borderRadius: '8px',
              fontSize: '0.9rem',
              border: '1px solid #fca5a5'
            }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          <div className="panels-container">
            <AnswerPanel messages={messages} isLoading={isLoading} />
            <ChartPanel response={response} isLoading={isLoading} />
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
