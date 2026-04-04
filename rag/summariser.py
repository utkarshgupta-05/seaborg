import pandas as pd


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
    temp_c = float(row_data["temp_c"])
    salinity = float(row_data["salinity"])
    depth_m = float(row_data["depth_m"])
    date = pd.to_datetime(row_data["date"]).strftime("%Y-%m-%d")
    lat = float(row_data["latitude"])
    lon = float(row_data["longitude"])

    return (
        f"Float {float_id} recorded a temperature of {temp_c:.1f}°C and salinity of "
        f"{salinity:.2f} PSU at {round(depth_m):.0f}m depth on {date} at coordinates "
        f"({lat:.2f}, {lon:.2f})."
    )
