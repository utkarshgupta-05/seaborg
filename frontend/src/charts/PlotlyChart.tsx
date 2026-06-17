import React from 'react';
import Plot from 'react-plotly.js';

interface PlotlyChartProps {
  visualizationData: any;
}

export const PlotlyChart: React.FC<PlotlyChartProps> = ({ visualizationData }) => {
  if (!visualizationData) {
    return null;
  }

  try {
    // visualizationData should be the output of fig.to_plotly_json()
    // It typically has { data: [...], layout: {...} }
    const { data, layout } = visualizationData;

    if (!data) {
      throw new Error("Invalid Plotly payload: missing 'data'");
    }

    // Default layout adjustments for responsive rendering within our React UI
    const mergedLayout = {
      ...layout,
      autosize: true,
      margin: { t: 40, b: 40, l: 40, r: 40, ...layout?.margin },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {
        family: "Inter, sans-serif",
        ...layout?.font
      }
    };

    return (
      <div style={{ width: '100%', height: '100%', minHeight: '350px' }}>
        <Plot
          data={data}
          layout={mergedLayout}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ responsive: true, displayModeBar: false }}
        />
      </div>
    );
  } catch (error) {
    console.error("Plotly render error:", error);
    return (
      <div style={{ 
        padding: '24px', 
        color: '#ef4444', 
        backgroundColor: '#fee2e2', 
        borderRadius: '8px',
        textAlign: 'center' 
      }}>
        Chart rendering failed due to malformed data payload.
      </div>
    );
  }
};
