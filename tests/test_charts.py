import os
import shutil
import pandas as pd
import numpy as np
import pytest
import plotly.graph_objects as go

from visualisation.map_chart import plot_float_map
from visualisation.profile_chart import plot_depth_profile
from visualisation.timeseries_chart import plot_timeseries
from visualisation.exporter import export_csv, export_chart_html, export_chart_png

def get_test_dataframe():
    # Attempt to load actual parquet file
    parquet_path = "data/processed/argo.parquet"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    
    # Fallback dummy DataFrame for standalone testing
    print("[TEST] Using dummy DataFrame for testing")
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "id": range(1, 101),
        "float_id": ["D13857"] * 50 + ["D99999"] * 50,
        "date": dates.tolist(),
        "latitude": np.random.uniform(-90.0, 90.0, 100),
        "longitude": np.random.uniform(-180.0, 180.0, 100),
        "depth_m": np.random.uniform(0.0, 2000.0, 100),
        "temp_c": np.random.uniform(2.0, 30.0, 100),
        "salinity": np.random.uniform(33.0, 37.0, 100),
        "oxygen": np.random.uniform(100.0, 300.0, 100)
    })

def test_charts_nominal():
    df = get_test_dataframe()
    first_float = df["float_id"].iloc[0]

    # Map chart
    fig1 = plot_float_map(df.head(50))
    assert fig1 is not None
    assert isinstance(fig1, go.Figure)

    # Depth profile chart
    fig2 = plot_depth_profile(df, first_float, "temp_c")
    assert fig2 is not None
    assert isinstance(fig2, go.Figure)

    # Timeseries chart
    fig3 = plot_timeseries(df, first_float, "temp_c")
    assert fig3 is not None
    assert isinstance(fig3, go.Figure)

    print("All charts: OK")

def test_charts_edge_cases():
    df = get_test_dataframe()
    first_float = df["float_id"].iloc[0]

    # Test empty DataFrame
    empty_df = pd.DataFrame(columns=df.columns)
    fig_empty_map = plot_float_map(empty_df)
    assert fig_empty_map is not None
    
    fig_empty_profile = plot_depth_profile(empty_df, first_float)
    assert fig_empty_profile is not None
    
    fig_empty_ts = plot_timeseries(empty_df, first_float)
    assert fig_empty_ts is not None

    # Test missing columns (e.g. salinity)
    no_salinity_df = df.drop(columns=["salinity"], errors="ignore")
    fig_no_sal = plot_float_map(no_salinity_df)
    assert fig_no_sal is not None
    
    # Test missing oxygen
    no_oxygen_df = df.drop(columns=["oxygen"], errors="ignore")
    fig_no_oxy = plot_depth_profile(no_oxygen_df, first_float, "oxygen")
    assert fig_no_oxy is not None

def test_exporters():
    df = get_test_dataframe()
    first_float = df["float_id"].iloc[0]
    
    fig = plot_depth_profile(df, first_float, "temp_c")

    csv_path = export_csv(df.head(10), "test_export.csv")
    html_path = export_chart_html(fig, "test_export.html")
    png_path = export_chart_png(fig, "test_export.png")

    assert os.path.exists(csv_path)
    assert os.path.exists(html_path)
    assert os.path.exists(png_path)

    # Clean up test export files
    for path in [csv_path, html_path, png_path]:
        if os.path.exists(path):
            os.remove(path)

    print("Exports: OK")
