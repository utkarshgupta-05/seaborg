import React from 'react';
import { FaChartLine, FaFileAlt, FaDownload } from 'react-icons/fa';
import { PlotlyChart } from '../charts/PlotlyChart';
import { ChatResponse } from '../types/api';

interface ChartPanelProps {
  response: ChatResponse | null;
  isLoading: boolean;
}

export const ChartPanel: React.FC<ChartPanelProps> = ({ response, isLoading }) => {
  if (isLoading) {
    return (
      <div className="card" style={{ gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent)' }}>
            <FaChartLine size={18} />
            <h2 style={{ fontSize: '1.1rem', margin: 0 }}>Chart Panel</h2>
          </div>
          <div style={{ display: 'flex', gap: '12px', color: 'var(--text-muted)' }}>
            <FaFileAlt size={14} />
            <FaDownload size={14} />
          </div>
        </div>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          height: '350px',
          background: '#f8fafc',
          borderRadius: '8px'
        }}>
          <span className="spinner" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent', width: '30px', height: '30px' }} />
        </div>
      </div>
    );
  }

  const hasChart = response && response.visualization_data;

  return (
    <div className="card" style={{ gap: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent)' }}>
          <FaChartLine size={18} />
          <h2 style={{ fontSize: '1.1rem', margin: 0 }}>Chart Panel</h2>
        </div>
        <div style={{ display: 'flex', gap: '12px', color: 'var(--text-muted)' }}>
          <FaFileAlt size={14} style={{ cursor: 'pointer' }} />
          <FaDownload size={14} style={{ cursor: 'pointer' }} />
        </div>
      </div>

      {!hasChart ? (
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          height: '350px',
          background: '#f8fafc',
          borderRadius: '8px',
          color: 'var(--text-muted)',
          fontSize: '0.9rem'
        }}>
          No visualization available
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          {response.chart_title && (
            <h3 style={{ fontSize: '0.9rem', marginBottom: '8px', color: 'var(--text-main)', textAlign: 'center' }}>
              {response.chart_title}
            </h3>
          )}
          <PlotlyChart visualizationData={response.visualization_data} />
          {response.chart_description && (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '8px' }}>
              {response.chart_description}
            </p>
          )}
        </div>
      )}
    </div>
  );
};
