import React from 'react';
import { FaAnchor, FaCheckCircle, FaTimesCircle } from 'react-icons/fa';

interface HeaderProps {
  isBackendConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({ isBackendConnected }) => {
  return (
    <header className="header-area" style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'space-between',
      padding: '0 24px',
      backgroundColor: 'var(--bg-dark)',
      color: 'white',
      height: '64px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <FaAnchor size={24} color="var(--accent)" />
        <h1 style={{ fontSize: '1.25rem', fontWeight: 600, margin: 0 }}>SeaBorg</h1>
      </div>
      
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '8px', 
        fontSize: '0.875rem',
        backgroundColor: 'rgba(255,255,255,0.1)',
        padding: '6px 12px',
        borderRadius: '20px'
      }}>
        {isBackendConnected ? (
          <>
            <FaCheckCircle color="#10b981" />
            <span>Backend Connected</span>
          </>
        ) : (
          <>
            <FaTimesCircle color="#ef4444" />
            <span>Backend Offline</span>
          </>
        )}
      </div>
    </header>
  );
};
