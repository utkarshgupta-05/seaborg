import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

VARIABLE_LABELS = {
    "temp_c": "Temperature (°C)",
    "salinity": "Salinity (PSU)",
    "oxygen": "Oxygen",
}

VARIABLE_TITLES = {
    "temp_c": "Temperature",
    "salinity": "Salinity",
    "oxygen": "Oxygen",
}

def empty_figure(message: str = "No data available") -> go.Figure:
    """Returns a standardized empty Plotly figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig
