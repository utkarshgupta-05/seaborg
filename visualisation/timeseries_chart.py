import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

from .common import empty_figure, VARIABLE_LABELS, VARIABLE_TITLES

logger = logging.getLogger(__name__)

def plot_timeseries(df: pd.DataFrame, float_id: str, variable: str = "temp_c") -> go.Figure:
    """
    Plots daily or monthly averages of a variable over time for one float.
    
    Args:
        df: DataFrame containing the readings.
        float_id: Unique float identifier.
        variable: One of 'temp_c', 'salinity', 'oxygen'.
    """
    var_title = VARIABLE_TITLES.get(variable, variable.capitalize())
    var_label = VARIABLE_LABELS.get(variable, variable.capitalize())
    
    if df is None or df.empty or "float_id" not in df.columns or "date" not in df.columns or variable not in df.columns:
        logger.debug("[VIZ] Timeseries chart generated (empty)")
        return empty_figure()

    # Filter by float_id
    float_df = df[df["float_id"] == float_id].copy()
    
    # Handle NaNs
    float_df = float_df.dropna(subset=["date", variable])
    
    if float_df.empty:
        logger.debug("[VIZ] Timeseries chart generated (empty)")
        return empty_figure()

    # Convert date column to datetime
    float_df["date"] = pd.to_datetime(float_df["date"])
    
    # Check count of unique dates
    unique_dates = float_df["date"].dt.date.nunique()
    
    if unique_dates > 90:
        # Aggregate to monthly averages
        # To avoid problems with Resample on non-index or non-unique, we do groupby Period or resample.
        # Let's set 'date' as index, resample to start of month 'MS' and compute mean.
        float_df = float_df.set_index("date")
        aggregated = float_df[[variable]].resample("MS").mean().reset_index()
        title_suffix = " (Monthly Average)"
    else:
        # Aggregate to daily averages
        float_df = float_df.set_index("date")
        aggregated = float_df[[variable]].resample("D").mean().reset_index()
        title_suffix = " (Daily Average)"

    # Drop any NaNs created by resampling empty periods
    aggregated = aggregated.dropna(subset=[variable])
    
    if aggregated.empty:
        logger.debug("[VIZ] Timeseries chart generated (empty)")
        return empty_figure()

    # Sort to ensure line plots chronologically
    aggregated = aggregated.sort_values(by="date")

    fig = go.Figure()
    
    # Format dates as YYYY-MM-DD for text hover
    hover_texts = []
    for _, row in aggregated.iterrows():
        dt = row["date"]
        if hasattr(dt, "strftime"):
            dt_str = dt.strftime("%Y-%m-%d")
        else:
            dt_str = str(dt)
        hover_texts.append(
            f"Float: {float_id}<br>Date: {dt_str}<br>{var_label}: {row[variable]:.3f}"
        )

    fig.add_trace(
        go.Scatter(
            x=aggregated["date"],
            y=aggregated[variable],
            mode="lines+markers",
            name=var_label,
            text=hover_texts,
            hoverinfo="text"
        )
    )

    fig.update_layout(
        title=f"{var_title} Over Time — Float {float_id}{title_suffix}",
        xaxis_title="Date",
        yaxis_title=var_label,
        hovermode="closest"
    )

    logger.debug("[VIZ] Timeseries chart generated")
    return fig
