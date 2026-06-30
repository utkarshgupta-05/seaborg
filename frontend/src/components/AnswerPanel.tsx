import React from 'react';
import { FaRobot, FaUser } from 'react-icons/fa';
import { ChatResponse } from '../types/api';
import { MessagePair } from '../hooks/useChat';

interface AnswerPanelProps {
  messages: MessagePair[];
  isLoading: boolean;
}

export const AnswerPanel: React.FC<AnswerPanelProps> = ({ messages, isLoading }) => {
  if (messages.length === 0 && !isLoading) {
    return (
      <div className="card" style={{ gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent)' }}>
          <FaRobot size={18} />
          <h2 style={{ fontSize: '1.1rem', margin: 0 }}>Answer Panel</h2>
        </div>
        <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '12px' }}>
          Your conversation transcript will appear here.
        </div>
      </div>
    );
  }

  const renderMetadata = (response: ChatResponse) => {
    const items: { label: string; value: React.ReactNode }[] = [];
    if (response.float_ids && response.float_ids.length > 0) {
      items.push({ label: 'Floats Analyzed', value: String(response.float_ids.length) });
    }
    if (response.confidence) {
      items.push({ label: 'Confidence', value: `${(response.confidence * 100).toFixed(0)}%` });
    }
    if (response.metadata) {
      Object.entries(response.metadata).forEach(([key, val]) => {
        const formattedKey = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        let displayVal: React.ReactNode = String(val);
        if (key === 'query_type') {
          const qt = String(val).toLowerCase();
          let bgColor = '#cbd5e1';
          let textColor = '#334155';
          let label = qt;
          if (qt === 'structured') { bgColor = '#bfdbfe'; textColor = '#1e3a8a'; label = 'Structured Query'; }
          else if (qt === 'semantic') { bgColor = '#e9d5ff'; textColor = '#581c87'; label = 'Semantic Retrieval'; }
          else if (qt === 'hybrid') { bgColor = '#bbf7d0'; textColor = '#14532d'; label = 'Hybrid'; }
          
          displayVal = (
            <span style={{
              backgroundColor: bgColor, color: textColor, padding: '2px 8px', borderRadius: '12px', fontSize: '0.8rem', fontWeight: 600
            }}>
              {label}
            </span>
          );
        } else if (typeof val === 'object' && val !== null) {
          displayVal = Object.entries(val).map(([k, v]) => `${k}: ${v}`).join(', ');
        }
        items.push({ label: formattedKey, value: displayVal });
      });
    }
    return items;
  };

  return (
    <div className="card" style={{ gap: '24px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent)', borderBottom: '1px solid #e2e8f0', paddingBottom: '12px' }}>
        <FaRobot size={18} />
        <h2 style={{ fontSize: '1.1rem', margin: 0 }}>Conversation Transcript</h2>
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
        {messages.map((msg, index) => {
          const metadataItems = msg.response ? renderMetadata(msg.response) : [];
          
          return (
            <div key={msg.id} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {/* User Bubble */}
              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
                <div style={{ backgroundColor: '#f1f5f9', padding: '12px 16px', borderRadius: '16px 16px 0 16px', maxWidth: '80%', color: '#1e293b' }}>
                  <strong>{msg.query}</strong>
                </div>
                <div style={{ color: '#94a3b8', marginTop: '4px' }}><FaUser size={16} /></div>
              </div>

              {/* AI Bubble */}
              <div style={{ display: 'flex', gap: '12px' }}>
                <div style={{ color: 'var(--accent)', marginTop: '4px' }}><FaRobot size={18} /></div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%' }}>
                  {msg.isLoading ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '8px' }}>
                      <div style={{ height: '16px', background: '#e2e8f0', borderRadius: '4px', width: '90%' }}></div>
                      <div style={{ height: '16px', background: '#e2e8f0', borderRadius: '4px', width: '80%' }}></div>
                      <div style={{ height: '16px', background: '#e2e8f0', borderRadius: '4px', width: '85%' }}></div>
                    </div>
                  ) : msg.error ? (
                    <div style={{ color: 'red' }}>{msg.error}</div>
                  ) : msg.response ? (
                    <>
                      <div style={{ fontSize: '1rem', whiteSpace: 'pre-wrap', lineHeight: '1.6', color: 'var(--text-main)' }}>
                        {msg.response.answer}
                      </div>

                      {metadataItems.length > 0 && (
                        <div style={{
                          marginTop: '8px',
                          backgroundColor: '#f8fafc',
                          borderRadius: '8px',
                          padding: '12px',
                          border: '1px solid #e2e8f0'
                        }}>
                          <h3 style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                            Query Metadata
                          </h3>
                          <div style={{ display: 'grid', gap: '6px' }}>
                            {metadataItems.map((item, idx) => (
                              <div key={idx} style={{ display: 'flex', gap: '8px', fontSize: '0.9rem' }}>
                                <span style={{ fontWeight: 600, color: 'var(--bg-dark)', minWidth: '130px' }}>{item.label}:</span>
                                <span style={{ color: 'var(--text-muted)' }}>{item.value}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : null}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
