import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

from .common import empty_figure, VARIABLE_LABELS

logger = logging.getLogger(__name__)

def plot_float_map(df: pd.DataFrame, title: str | None = None, variable: str = "temp_c") -> go.Figure:
    """
    Generates a geospatial scatter map of float positions coloured by the specified variable.
    
    Args:
        df: DataFrame with columns: latitude, longitude, and optionally other variables.
        title: Optional custom chart title.
        variable: The variable to use for coloring the points (default 'temp_c').
    
    Returns:
        go.Figure: Plotly figure object.
    """
    required_cols = ["latitude", "longitude"]
    
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        logger.debug("[VIZ] Map chart generated (empty)")
        return empty_figure()
    chart_title = title if title else "ARGO Float Positions"
    
    # Ensure no NaN positions
    clean_df = df.dropna(subset=["latitude", "longitude"])
    if clean_df.empty:
        logger.debug("[VIZ] Map chart generated (empty)")
        return empty_figure()
    
    # Build hover_data dynamically
    hover_cols = ["float_id", "date", "depth_m", variable]
    hover_data = {col: True for col in hover_cols if col in clean_df.columns}
    hover_data["latitude"] = True
    hover_data["longitude"] = True
    
    color_col = variable if variable in clean_df.columns else None
    
    fig = px.scatter_geo(
        clean_df,
        lat="latitude",
        lon="longitude",
        color=color_col,
        color_continuous_scale="RdBu_r",
        title=chart_title,
        hover_data=hover_data,
        labels={variable: VARIABLE_LABELS.get(variable, variable)} if color_col else {}
    )
    
    fig.update_geos(
        showland=True,
        landcolor="rgb(243, 243, 243)",
        showocean=True,
        oceancolor="lightblue",
        showcountries=True,
        countrycolor="gray",
        showcoastlines=True,
        coastlinecolor="gray",
        projection_type="equirectangular"
    )
    
    logger.debug("[VIZ] Map chart generated")
    return fig
