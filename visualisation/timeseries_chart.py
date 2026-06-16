import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_timeseries(df: pd.DataFrame, float_id: str, variable: str = "temp_c") -> go.Figure:
    """
    Plots daily or monthly averages of a variable over time for one float.
    
    Args:
        df: DataFrame containing the readings.
        float_id: Unique float identifier.
        variable: One of 'temp_c', 'salinity', 'oxygen'.
    """
    friendly_labels = {
        "temp_c": "Temperature (°C)",
        "salinity": "Salinity (PSU)",
        "oxygen": "Oxygen"
    }
    var_title = {
        "temp_c": "Temperature",
        "salinity": "Salinity",
        "oxygen": "Oxygen"
    }.get(variable, variable.capitalize())
    
    var_label = friendly_labels.get(variable, variable.capitalize())
    
    if df is None or df.empty or "float_id" not in df.columns or "date" not in df.columns or variable not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        print("[VIZ] Timeseries chart generated")
        return fig

    # Filter by float_id
    float_df = df[df["float_id"] == float_id].copy()
    
    # Handle NaNs
    float_df = float_df.dropna(subset=["date", variable])
    
    if float_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        print("[VIZ] Timeseries chart generated")
        return fig

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
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        print("[VIZ] Timeseries chart generated")
        return fig

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

    print("[VIZ] Timeseries chart generated")
    return fig
