TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_float_data",
            "description": (
                "Retrieve all readings for a specific ARGO float "
                "within an optional date range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "float_id": {
                        "type": "string",
                        "description": "The unique ARGO float identifier.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional).",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional).",
                    },
                },
                "required": ["float_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearby_floats",
            "description": (
                "Find all ARGO floats within a given radius "
                "of a latitude/longitude coordinate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude in decimal degrees.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude in decimal degrees.",
                    },
                    "radius_km": {
                        "type": "number",
                        "description": "Search radius in kilometres.",
                    },
                },
                "required": ["latitude", "longitude", "radius_km"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_parameter_stats",
            "description": (
                "Return summary statistics (mean, min, max, std) "
                "for a parameter across a named region."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "enum": ["temp_c", "salinity", "oxygen"],
                        "description": "The oceanographic parameter to summarise.",
                    },
                    "region_name": {
                        "type": "string",
                        "description": "Name of the ocean region (e.g. Indian Ocean).",
                    },
                    "depth_min": {
                        "type": "number",
                        "description": "Minimum depth in metres (optional).",
                    },
                    "depth_max": {
                        "type": "number",
                        "description": "Maximum depth in metres (optional).",
                    },
                },
                "required": ["parameter", "region_name"],
            },
        },
    },
]