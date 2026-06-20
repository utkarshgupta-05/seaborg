import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

def _get_export_path(filename: str, ext: str) -> str:
    # Resolve absolute path to data/exports/ in workspace
    # The parent of 'visualisation' is the workspace root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    export_dir = os.path.join(base_dir, "data", "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Clean filename of extension if specified
    name_only, _ = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{name_only}_{timestamp}{ext}"
    return os.path.join(export_dir, full_name)

def export_csv(df: pd.DataFrame, filename: str) -> str:
    path = _get_export_path(filename, ".csv")
    df.to_csv(path, index=False)
    logger.debug("[EXPORT] CSV saved")
    return path

def export_chart_html(fig: go.Figure, filename: str) -> str:
    path = _get_export_path(filename, ".html")
    fig.write_html(path)
    logger.debug("[EXPORT] HTML saved")
    return path

def export_chart_png(fig: go.Figure, filename: str) -> str:
    path = _get_export_path(filename, ".png")
    try:
        fig.write_image(path, engine="kaleido")
        logger.debug("[EXPORT] PNG saved")
        return path
    except Exception as e:
        logger.warning(f"[EXPORT] PNG export failed: {e}. Graceful fallback applied.")
        # Fallback: write text description/placeholder
        fallback_path = path + ".txt"
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(f"PNG export failed/unavailable.\nError detail: {e}\nFigure Title: {fig.layout.title.text if fig.layout.title else 'None'}\n")
        return fallback_path
