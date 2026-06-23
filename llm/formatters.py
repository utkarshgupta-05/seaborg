import pandas as pd
from schema.variables import DEFAULT_VARIABLE

def format_row(row: pd.Series, variable: str = DEFAULT_VARIABLE) -> str:
    """
    Formats a single row into a bullet string parameterized by the requested variable.
    Handles NaN values gracefully.
    """
    depth = f"{row.get('depth_m', float('nan')):.0f}m" if pd.notna(row.get('depth_m')) else "N/A"
    temp = f"{row.get('temp_c', float('nan')):.1f}°C" if pd.notna(row.get('temp_c')) else "N/A"
    sal = f"{row.get('salinity', float('nan')):.2f} PSU" if pd.notna(row.get('salinity')) else "N/A"
    oxy = f"{row.get('oxygen', float('nan')):.2f}" if pd.notna(row.get('oxygen')) else "N/A"

    float_id = row.get('float_id', 'Unknown')
    date_val = row.get('date', 'Unknown')

    if variable == "salinity":
        return f"• Float {float_id} | {date_val} | Depth: {depth} | Salinity: {sal} | (Temp: {temp})"
    elif variable == "oxygen":
        return f"• Float {float_id} | {date_val} | Depth: {depth} | Oxygen: {oxy} | (Temp: {temp})"
    elif variable == "depth_m":
        return f"• Float {float_id} | {date_val} | Depth: {depth} | (Temp: {temp})"
    else:
        return f"• Float {float_id} | {date_val} | Depth: {depth} | Temp: {temp} | (Salinity: {sal})"
