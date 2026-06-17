import React, { useState } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

interface SearchBarProps {
  onSend: (message: string) => void;
  isLoading: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSend, isLoading }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (input.trim() && !isLoading) {
      onSend(input);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  const handleExampleClick = (query: string) => {
    setInput(query);
    onSend(query);
  };

  const examples = [
    "Temperature at 500m",
    "Atlantic Ocean Temperature",
    "Show Temperature Profile",
    "Show Float Location"
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <form 
        onSubmit={handleSubmit}
        style={{
          display: 'flex',
          backgroundColor: 'var(--card-bg)',
          borderRadius: 'var(--border-radius)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
          padding: '6px',
          border: '1px solid #e2e8f0',
          position: 'relative'
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. show temperature at 500m in atlantic ocean"
          disabled={isLoading}
          style={{
            flex: 1,
            border: 'none',
            outline: 'none',
            padding: '12px 16px',
            fontSize: '1rem',
            color: 'var(--text-main)',
            backgroundColor: 'transparent'
          }}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          style={{
            backgroundColor: 'var(--accent)',
            color: 'white',
            borderRadius: '8px',
            padding: '0 20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: (isLoading || !input.trim()) ? 0.6 : 1,
            transition: 'background 0.2s',
          }}
        >
          {isLoading ? (
            <span className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
          ) : (
            <FaPaperPlane size={16} />
          )}
        </button>
      </form>

      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', paddingLeft: '4px' }}>
        {examples.map((ex, i) => (
          <button
            key={i}
            onClick={() => handleExampleClick(ex)}
            disabled={isLoading}
            style={{
              fontSize: '0.8rem',
              padding: '6px 12px',
              backgroundColor: 'var(--card-bg)',
              color: 'var(--text-main)',
              border: '1px solid #e2e8f0',
              borderRadius: '16px',
              transition: 'all 0.2s',
              opacity: isLoading ? 0.6 : 1,
            }}
            onMouseOver={(e) => {
              if(!isLoading) {
                e.currentTarget.style.borderColor = 'var(--accent)';
                e.currentTarget.style.color = 'var(--accent)';
              }
            }}
            onMouseOut={(e) => {
              if(!isLoading) {
                e.currentTarget.style.borderColor = '#e2e8f0';
                e.currentTarget.style.color = 'var(--text-main)';
              }
            }}
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
};
