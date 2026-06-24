import pandas as pd
from schema.variables import DEFAULT_VARIABLE

_FMT = {
    "temp_c": ("°C", "Temp", ".1f"),
    "salinity": (" PSU", "Salinity", ".2f"),
    "oxygen": ("", "Oxygen", ".2f"),
    "chlorophyll": ("", "Chlorophyll", ".2f"),
    "nitrate": ("", "Nitrate", ".2f"),
}

def format_row(row: pd.Series, variable: str = DEFAULT_VARIABLE) -> str:
    """
    Formats a single row into a bullet string parameterized by the requested variable.
    Handles NaN values gracefully using a table-driven approach.
    """
    depth = f"{row.get('depth_m', float('nan')):.0f}m" if pd.notna(row.get('depth_m')) else "N/A"
    
    vals = {}
    for k, (unit, title, fmt) in _FMT.items():
        val = row.get(k, float('nan'))
        if pd.notna(val):
            vals[k] = f"{val:{fmt}}{unit}"
        else:
            vals[k] = "N/A"

    float_id = row.get('float_id', 'Unknown')
    date_val = row.get('date', 'Unknown')

    base = f"• Float {float_id} | {date_val} | Depth: {depth}"

    if variable == "depth_m":
        return f"{base} | (Temp: {vals['temp_c']})"
    
    # Lead with requested variable
    if variable in _FMT:
        _, title, _ = _FMT[variable]
        base += f" | {title}: {vals[variable]}"
        
    # Secondary context
    if variable == "temp_c":
        base += f" | (Salinity: {vals['salinity']})"
    else:
        base += f" | (Temp: {vals['temp_c']})"
        
    return base
