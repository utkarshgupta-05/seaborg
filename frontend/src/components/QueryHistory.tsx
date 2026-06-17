import React, { useEffect, useState } from 'react';
import { FaHistory, FaChevronRight } from 'react-icons/fa';

interface QueryHistoryProps {
  onRerun: (query: string) => void;
  newQueryAdded?: string | null;
}

const HISTORY_KEY = 'seaborg_query_history';
const MAX_HISTORY = 10;

export const QueryHistory: React.FC<QueryHistoryProps> = ({ onRerun, newQueryAdded }) => {
  const [history, setHistory] = useState<string[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem(HISTORY_KEY);
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse history");
      }
    }
  }, []);

  useEffect(() => {
    if (newQueryAdded) {
      setHistory(prev => {
        // Remove if it exists to move it to the top
        const filtered = prev.filter(q => q !== newQueryAdded);
        const updated = [newQueryAdded, ...filtered].slice(0, MAX_HISTORY);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
        return updated;
      });
    }
  }, [newQueryAdded]);

  if (history.length === 0) {
    return (
      <div style={{ padding: '16px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
        No recent queries.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', marginBottom: '8px', paddingLeft: '8px' }}>
        <FaHistory size={14} />
        <span style={{ fontSize: '0.85rem', fontWeight: 600, textTransform: 'uppercase' }}>Recent Queries</span>
      </div>
      {history.map((q, idx) => (
        <button
          key={idx}
          onClick={() => onRerun(q)}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '10px 12px',
            backgroundColor: 'transparent',
            borderRadius: '6px',
            textAlign: 'left',
            color: 'var(--text-main)',
            fontSize: '0.9rem',
            transition: 'background 0.2s',
            border: '1px solid transparent'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-light)';
            e.currentTarget.style.borderColor = '#e2e8f0';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.borderColor = 'transparent';
          }}
        >
          <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '85%' }}>{q}</span>
          <FaChevronRight size={10} color="var(--text-muted)" />
        </button>
      ))}
    </div>
  );
};
