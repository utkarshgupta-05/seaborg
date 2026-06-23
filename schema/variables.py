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
}

VARIABLE_LABELS = {k: v["label"] for k, v in VARIABLE_REGISTRY.items()}
VARIABLE_TITLES = {k: v["title"] for k, v in VARIABLE_REGISTRY.items()}

def detect_variable(question: str) -> Optional[str]:
    """
    Detects the requested variable in the question.
    Prioritizes exact whole-word matches over substring matches.
    Returns None if no variable is explicitly matched.
    """
    q_lower = question.lower()
    
    # Pre-compute alias mappings and sort by length descending to match longest first
    aliases_to_vars = []
    for var_key, metadata in VARIABLE_REGISTRY.items():
        for alias in metadata["aliases"]:
            aliases_to_vars.append((alias, var_key))
            
    # Sort by length of alias (longest first)
    aliases_to_vars.sort(key=lambda x: len(x[0]), reverse=True)
    
    # 1. Exact whole-word match (confidence-based precedence)
    for alias, var_key in aliases_to_vars:
        # \b ensures word boundary
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, q_lower):
            return var_key
            

            
    return None
