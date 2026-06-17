import React from 'react';
import { FaPlus } from 'react-icons/fa';
import { QueryHistory } from './QueryHistory';

interface SidebarProps {
  onNewQuery: () => void;
  onRerunQuery: (query: string) => void;
  newQueryAdded?: string | null;
}

export const Sidebar: React.FC<SidebarProps> = ({ onNewQuery, onRerunQuery, newQueryAdded }) => {
  return (
    <aside className="sidebar-area">
      <div style={{ padding: '24px 16px 16px' }}>
        <button 
          onClick={onNewQuery}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            backgroundColor: 'var(--accent)',
            color: 'white',
            padding: '12px',
            borderRadius: 'var(--border-radius)',
            fontWeight: 500,
            transition: 'background 0.2s',
          }}
          onMouseOver={e => e.currentTarget.style.backgroundColor = 'var(--accent-hover)'}
          onMouseOut={e => e.currentTarget.style.backgroundColor = 'var(--accent)'}
        >
          <FaPlus size={14} />
          New Query
        </button>
      </div>
      
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <QueryHistory onRerun={onRerunQuery} newQueryAdded={newQueryAdded} />
      </div>

      <div style={{ padding: '16px', borderTop: '1px solid #e2e8f0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        <p style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>About SeaBorg</p>
        <p>AI-Powered Ocean Data Explorer.</p>
        <p style={{ marginTop: '4px' }}>Analyze ARGO float data via semantic search and structured queries.</p>
      </div>
    </aside>
  );
};
