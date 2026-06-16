# Phase 5 — Visualization Engine

This document outlines the architecture and implementation of the SeaBorg **Visualization Engine**, designed specifically to generate interactive Plotly visualizations that are directly serializable into React-compatible JSON structures.

## 1. Design & React-Ready Architecture
Unlike standard Python applications that render HTML or raw figures on the backend, SeaBorg's visualization engine has been built for a decoupled **React + TypeScript frontend** (Phase 6). 
Instead of returning complex figure objects or rendering state on the server:
1. The backend builds Plotly figures (`go.Figure`) using `plotly.graph_objects` and `plotly.express`.
2. It calls `fig.to_plotly_json()`.
3. The resulting structure is processed by a recursive sanitization helper `sanitize_plotly_json(obj)` that cleans and translates non-JSON-serializable Python/Pandas types (e.g. Pandas Timestamps, datetime objects, NumPy scalars, `NaN` or `Infinite` floats) into clean, JSON-serializable primitives (standard floats/ints, strings, or `None`).
4. The sanitized payload is attached directly to the `visualization_data` field in `ChatResponse`, allowing the React frontend (using `react-plotly.js` or `plotly.js`) to render the chart immediately without any client-side parsing.

---

## 2. Visualization Components

### Map Chart (`visualisation/map_chart.py`)
- **Function:** `plot_float_map(df, title=None)`
- **Behavior:** Renders a Plotly `scatter_geo` map showing float positions colored by temperature using the reversed Red-Blue (`RdBu_r`) color scale. Points include tooltips with date, latitude, longitude, depth, temperature, and salinity. The geos layout is customized to explicitly show the land masses, blue oceans, and country borders.

### Profile Chart (`visualisation/profile_chart.py`)
- **Function:** `plot_depth_profile(df, float_id, variable="temp_c")`
- **Behavior:** Renders a line+marker depth profile for a specific float. To conform to oceanographic standards, the **Y-axis is inverted** (depth=0 at the top, increasing downwards). Variables (`temp_c`, `salinity`, `oxygen`) are mapped to user-friendly titles and units. NaNs and empty states are gracefully caught and returned as clean fallback charts.

### Time Series Chart (`visualisation/timeseries_chart.py`)
- **Function:** `plot_timeseries(df, float_id, variable="temp_c")`
- **Behavior:** Aggregates float readings over time. If the number of unique dates is **greater than 90**, it dynamically downsamples and resamples to **Monthly Averages** to avoid overcrowding. Otherwise, it plots **Daily Averages**. It plots a line+marker chart with timestamp strings.

---

## 3. Export System (`visualisation/exporter.py`)
Provides automated file writers saving to `data/exports/`:
- **`export_csv(df, filename)`**: Saves dataframes to CSV.
- **`export_chart_html(fig, filename)`**: Saves interactive HTML charts.
- **`export_chart_png(fig, filename)`**: Export static PNGs using the `kaleido` engine (integrated into `requirements.txt`). Implements a graceful fallback: if the system lacks `kaleido` dependencies or fails, it saves a detailed text description file `.png.txt` containing the chart information and doesn't crash the server.

All export paths automatically append the current timestamp (`_YYYYMMDD_HHMMSS`) to ensure uniqueness and prevent naming collisions.

---

## 4. API Integration & Routing (`api/routes/chat.py`)
The chat API uses a specialized visualization router to decide when to generate payloads:
- Ordinary queries like *"summarize this profile"* or *"explain salinity"* return `visualization_type=None` and `visualization_data=None` to keep payloads lightweight.
- Specific visualization requests (e.g. queries containing intent words like *show*, *plot*, *map*, *trend*, *over time*, *location*) invoke the matching Plotly engine, serialize the figure, and populate `visualization_type`, `visualization_data`, `chart_title`, and `chart_description`.
- Preserves backward compatibility by maintaining the existing `chart_type` keyword classifier.

---

## 5. Verification and Tests (`tests/test_charts.py`)
A comprehensive test suite validates:
1. Nominal chart generation for maps, profiles, and time-series.
2. Edge-case safety (empty DataFrames, missing optional columns like salinity or oxygen).
3. Exporter behavior, ensuring CSV, HTML, and PNG (with fallback) files are written correctly.
4. FastAPI endpoint validation to ensure the returned responses are fully JSON serializable and contain the correct fields.
