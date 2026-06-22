import pandas as pd
import plotly.graph_objects as go
import logging

from .common import empty_figure, VARIABLE_LABELS, VARIABLE_TITLES

logger = logging.getLogger(__name__)

def plot_depth_profile(df: pd.DataFrame, float_id: str, variable: str = "temp_c") -> go.Figure:
    """
    Plots depth vs variable profile chart for a single float.
    
    Args:
        df: DataFrame containing ocean profiles.
        float_id: Identifier of the float to plot.
        variable: Column name of the variable to plot.
    """
    var_title = VARIABLE_TITLES.get(variable, variable.capitalize())
    var_label = VARIABLE_LABELS.get(variable, variable.capitalize())
    
    if df is None or df.empty or "float_id" not in df.columns or "depth_m" not in df.columns:
        logger.debug("[VIZ] Profile chart generated (empty)")
        return empty_figure()

    if variable not in df.columns:
        logger.debug("[VIZ] Profile chart generated (missing variable)")
        return empty_figure("No data available for requested variable.")

    # Filter to float_id
    float_df = df[df["float_id"] == float_id].copy()
    
    # Handle NaNs in depth_m or variable
    float_df = float_df.dropna(subset=["depth_m", variable])
    
    if float_df.empty:
        logger.debug("[VIZ] Profile chart generated (empty)")
        return empty_figure()
    float_df = float_df.sort_values(by="depth_m", ascending=True)

    fig = go.Figure()
    
    # Set up hover text
    hover_texts = []
    for _, row in float_df.iterrows():
        dt = row.get("date", "N/A")
        if hasattr(dt, "strftime"):
            dt_str = dt.strftime("%Y-%m-%d")
        else:
            dt_str = str(dt)
        hover_texts.append(
            f"Float: {float_id}<br>Date: {dt_str}<br>Depth: {row['depth_m']} m<br>{var_label}: {row[variable]}"
        )

    fig.add_trace(
        go.Scatter(
            x=float_df[variable],
            y=float_df["depth_m"],
            mode="lines+markers",
            name=var_label,
            text=hover_texts,
            hoverinfo="text"
        )
    )

    fig.update_layout(
        title=f"{var_title} Profile — Float {float_id}",
        xaxis_title=var_label,
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
        hovermode="closest"
    )

    logger.debug("[VIZ] Profile chart generated")
    return fig
