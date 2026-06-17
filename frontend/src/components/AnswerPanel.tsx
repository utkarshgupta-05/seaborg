import React from 'react';
import { FaRobot } from 'react-icons/fa';
import { ChatResponse } from '../types/api';

interface AnswerPanelProps {
  response: ChatResponse | null;
  isLoading: boolean;
}

export const AnswerPanel: React.FC<AnswerPanelProps> = ({ response, isLoading }) => {
  if (isLoading) {
    return (
      <div className="card" style={{ gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent)' }}>
          <FaRobot size={18} />
          <h2 style={{ fontSize: '1.1rem', margin: 0 }}>Answer Panel</h2>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '8px' }}>
          <div style={{ height: '16px', background: '#e2e8f0', borderRadius: '4px', width: '90%' }}></div>
          <div style={{ height: '16px', background: '#e2e8f0', borderRadius: '4px', width: '80%' }}></div>
          <div style={{ height: '16px', background: '#e2e8f0', borderRadius: '4px', width: '85%' }}></div>
        </div>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="card" style={{ gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent)' }}>
          <FaRobot size={18} />
          <h2 style={{ fontSize: '1.1rem', margin: 0 }}>Answer Panel</h2>
        </div>
        <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '12px' }}>
          Your answer will appear here.
        </div>
      </div>
    );
  }

  const renderMetadata = () => {
    // Collect metadata items to render dynamically
    const items: { label: string; value: string }[] = [];
    
    // Core properties
    if (response.sql_used && response.sql_used !== "none" && response.sql_used !== "N/A (Structured Engine)") {
      items.push({ label: 'Query Type', value: 'Semantic Retrieval' });
    } else if (response.sql_used === "N/A (Structured Engine)") {
      items.push({ label: 'Query Type', value: 'Structured Query' });
    }

    if (response.float_ids && response.float_ids.length > 0) {
      items.push({ label: 'Floats Analyzed', value: String(response.float_ids.length) });
    }
    
    if (response.confidence) {
      items.push({ label: 'Confidence', value: `${(response.confidence * 100).toFixed(0)}%` });
    }

    // Dynamic metadata dictionary
    if (response.metadata) {
      Object.entries(response.metadata).forEach(([key, val]) => {
        // Format key (e.g., 'query_type' -> 'Query Type')
        const formattedKey = key
          .split('_')
          .map(w => w.charAt(0).toUpperCase() + w.slice(1))
          .join(' ');
        
        // Skip duplicate or hidden keys if necessary, or just render them all cleanly
        if (key !== 'query_type') { // We already handled general query type above
          let displayVal = String(val);
          if (typeof val === 'object' && val !== null) {
            displayVal = Object.entries(val).map(([k, v]) => `${k}: ${v}`).join(', ');
          }
          items.push({ label: formattedKey, value: displayVal });
        }
      });
    }

    return items;
  };

  const metadataItems = renderMetadata();

  return (
    <div className="card" style={{ gap: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent)' }}>
        <FaRobot size={18} />
        <h2 style={{ fontSize: '1.1rem', margin: 0 }}>Answer Panel</h2>
      </div>
      
      <div style={{ fontSize: '1rem', whiteSpace: 'pre-wrap', lineHeight: '1.6', color: 'var(--text-main)' }}>
        {response.answer}
      </div>

      {metadataItems.length > 0 && (
        <div style={{
          marginTop: '16px',
          backgroundColor: '#f8fafc',
          borderRadius: '8px',
          padding: '16px',
          border: '1px solid #e2e8f0'
        }}>
          <h3 style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.5px', 
            color: 'var(--text-muted)',
            marginBottom: '12px'
          }}>
            Query Metadata
          </h3>
          <div style={{ display: 'grid', gap: '8px' }}>
            {metadataItems.map((item, idx) => (
              <div key={idx} style={{ display: 'flex', gap: '8px', fontSize: '0.9rem' }}>
                <span style={{ fontWeight: 600, color: 'var(--bg-dark)', minWidth: '130px' }}>
                  {item.label}:
                </span>
                <span style={{ color: 'var(--text-muted)' }}>
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
