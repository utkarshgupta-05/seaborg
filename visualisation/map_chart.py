import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

from .common import empty_figure

logger = logging.getLogger(__name__)

def plot_float_map(df: pd.DataFrame, title: str | None = None) -> go.Figure:
    """
    Generates a geospatial scatter map of float positions coloured by temperature.
    
    Args:
        df: DataFrame with columns: latitude, longitude, temp_c, float_id, date, depth_m, salinity.
        title: Optional custom chart title.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    required_cols = ["latitude", "longitude", "temp_c", "float_id", "date", "depth_m", "salinity"]
    
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        logger.debug("[VIZ] Map chart generated (empty)")
        return empty_figure()
    chart_title = title if title else "ARGO Float Positions"
    
    # Ensure no NaN positions
    clean_df = df.dropna(subset=["latitude", "longitude"])
    if clean_df.empty:
        logger.debug("[VIZ] Map chart generated (empty)")
        return empty_figure()
    
    fig = px.scatter_geo(
        clean_df,
        lat="latitude",
        lon="longitude",
        color="temp_c",
        color_continuous_scale="RdBu_r",
        title=chart_title,
        hover_data={
            "float_id": True,
            "date": True,
            "latitude": True,
            "longitude": True,
            "depth_m": True,
            "temp_c": True,
            "salinity": True
        }
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
