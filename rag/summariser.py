import pandas as pd
from llm.geo_mapping import detect_region_from_coords

def summarise_row(row: dict | pd.Series) -> str:
    """
    Converts one ARGO data row into a fixed-format English sentence.

    Args:
        row: A dict-like or pandas Series containing SeaBorg profile fields.

    Returns:
        A single sentence in the exact required format for embedding.

    Side effects:
        None.
    """
    row_data = dict(row) if isinstance(row, pd.Series) else row

    float_id = str(row_data["float_id"])
    depth_m = float(row_data["depth_m"])
    date = pd.to_datetime(row_data["date"]).strftime("%Y-%m-%d")
    lat = float(row_data["latitude"])
    lon = float(row_data["longitude"])

    region = detect_region_from_coords(lat, lon)
    region_text = f" in the {region}" if region else ""
    
    measurements = []
    
    val = row_data.get("temp_c")
    if pd.notna(val): measurements.append(f"a temperature of {float(val):.1f}°C")
        
    val = row_data.get("salinity")
    if pd.notna(val): measurements.append(f"salinity of {float(val):.2f} PSU")
        
    val = row_data.get("oxygen")
    if pd.notna(val): measurements.append(f"oxygen of {float(val):.2f}")
        
    val = row_data.get("chlorophyll")
    if pd.notna(val): measurements.append(f"chlorophyll of {float(val):.2f}")
        
    val = row_data.get("nitrate")
    if pd.notna(val): measurements.append(f"nitrate of {float(val):.2f}")

    if len(measurements) > 1:
        measurements_str = ", ".join(measurements[:-1]) + f" and {measurements[-1]}"
    elif len(measurements) == 1:
        measurements_str = measurements[0]
    else:
        measurements_str = "no measurements"

    return (
        f"Float {float_id} recorded {measurements_str} at {round(depth_m):.0f}m depth{region_text} on {date} at coordinates "
        f"({lat:.2f}, {lon:.2f})."
    )
