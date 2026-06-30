import re
from typing import Optional

DEFAULT_VARIABLE = "temp_c"

VARIABLE_REGISTRY = {
    "temp_c": {
        "label": "Temperature (°C)",
        "title": "Temperature",
        "aliases": ["temperature", "temp", "thermal", "warmth", "heat"],
    },
    "salinity": {
        "label": "Salinity (PSU)",
        "title": "Salinity",
        "aliases": ["salinity", "salt", "saltiness", "salt level", "salt content"],
    },
    "depth_m": {
        "label": "Depth / Pressure (m)",
        "title": "Depth",
        "aliases": ["pressure", "pres", "dbar", "depth"],
    },
    "oxygen": {
        "label": "Oxygen",
        "title": "Oxygen",
        "aliases": ["oxygen", "o2", "dissolved oxygen", "oxygen level", "oxygen content"],
    },
    "chlorophyll": {
        "label": "Chlorophyll",
        "title": "Chlorophyll",
        "aliases": ["chlorophyll", "chl", "chla"],
    },
    "nitrate": {
        "label": "Nitrate",
        "title": "Nitrate",
        "aliases": ["nitrate", "no3"],
    },
}

VARIABLE_LABELS = {k: v["label"] for k, v in VARIABLE_REGISTRY.items()}
VARIABLE_TITLES = {k: v["title"] for k, v in VARIABLE_REGISTRY.items()}

# Pre-compute alias mappings and sort by length descending to match longest first
_ALIASES_TO_VARS = []
for var_key, metadata in VARIABLE_REGISTRY.items():
    for alias in metadata["aliases"]:
        _ALIASES_TO_VARS.append((alias, var_key))
        
# Sort by length of alias (longest first)
_ALIASES_TO_VARS.sort(key=lambda x: len(x[0]), reverse=True)

# Pre-compile regex patterns for efficiency
_ALIAS_PATTERNS = [(re.compile(r'\b' + re.escape(alias) + r'\b'), var_key) for alias, var_key in _ALIASES_TO_VARS]


def detect_variable(question: str) -> Optional[str]:
    """
    Detects the requested variable in the question.
    Prioritizes exact whole-word matches over substring matches.
    Returns None if no variable is explicitly matched.
    """
    q_lower = question.lower()
    
    # 1. Exact whole-word match (confidence-based precedence)
    for pattern, var_key in _ALIAS_PATTERNS:
        if pattern.search(q_lower):
            return var_key
            

            
    return None

def has_variable_data(df, variable: str) -> bool:
    """
    Returns True if `variable` is a real column in `df` with at least one
    non-null value. Use this on already-retrieved/filtered rows — it is
    the per-query complement to is_variable_available()'s dataset-wide check.
    """
    return variable in df.columns and df[variable].notna().any()
