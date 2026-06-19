import pytest

from structured_query.parser import parse_query

def test_depth_exact_formats():
    """Test exact/around depth variations (±50m window)."""
    queries = [
        "temperature at 500m",
        "salinity around 500 meters",
        "oxygen near 500 m",
        "500m",
    ]
    for q in queries:
        parsed = parse_query(q)
        assert parsed.depth_min == 450.0
        assert parsed.depth_max == 550.0
        assert parsed.metadata_filters["depth"] == "450–550m"

def test_depth_deeper_than():
    """Test deeper/below/greater than."""
    queries = [
        "deeper than 500m",
        "below 500 meters",
        "greater than 500 m",
    ]
    for q in queries:
        parsed = parse_query(q)
        assert parsed.depth_min == 500.0
        assert parsed.depth_max is None
        assert parsed.metadata_filters["depth"] == "500–∞m"

def test_depth_shallower_than():
    """Test shallower/above/less than."""
    queries = [
        "shallower than 100m",
        "above 100 meters",
        "less than 100 m",
    ]
    for q in queries:
        parsed = parse_query(q)
        assert parsed.depth_min is None
        assert parsed.depth_max == 100.0
        assert parsed.metadata_filters["depth"] == "0–100m"

def test_depth_between():
    """Test between ranges."""
    queries = [
        "between 200m and 400m",
        "between 200 and 400 meters",
        "between 200 to 400m",
    ]
    for q in queries:
        parsed = parse_query(q)
        assert parsed.depth_min == 200.0
        assert parsed.depth_max == 400.0
        assert parsed.metadata_filters["depth"] == "200–400m"

def test_depth_dash_or_to():
    """Test X-Ym and X to Y formats added in Phase 11."""
    queries = [
        "200-400m",
        "200 m to 400 m",
        "temperature 200-400 meters",
    ]
    for q in queries:
        parsed = parse_query(q)
        assert parsed.depth_min == 200.0
        assert parsed.depth_max == 400.0
        assert parsed.metadata_filters["depth"] == "200–400m"

def test_region_parsing():
    """Test detection of old and new regions."""
    regions = {
        "atlantic ocean": {"lat_min": -30, "lat_max": 30, "lon_min": -70, "lon_max": 20},
        "north sea": {"lat_min": 51, "lat_max": 61, "lon_min": -4, "lon_max": 10},
        "southern ocean": {"lat_min": -90, "lat_max": -60, "lon_min": -180, "lon_max": 180},
    }
    
    for name, bounds in regions.items():
        parsed = parse_query(f"average temp in the {name}")
        assert parsed.lat_min == bounds["lat_min"]
        assert parsed.lat_max == bounds["lat_max"]
        assert parsed.lon_min == bounds["lon_min"]
        assert parsed.lon_max == bounds["lon_max"]
        assert parsed.metadata_filters["region"] == name.title()

def test_combined_parsing():
    """Test combined depth and region queries."""
    q = "average salinity between 100m and 200m in the mediterranean sea"
    parsed = parse_query(q)
    
    assert parsed.depth_min == 100.0
    assert parsed.depth_max == 200.0
    assert parsed.metadata_filters["depth"] == "100–200m"
    
    assert parsed.metadata_filters["region"] == "Mediterranean Sea"
    assert parsed.lat_min == 30
    assert parsed.lat_max == 45
    assert parsed.lon_min == -5
    assert parsed.lon_max == 35

def test_empty_or_malformed_input():
    """Test parsing logic resilience."""
    parsed = parse_query("average temperature globally")
    assert not parsed.has_filters
    assert parsed.depth_min is None
    assert parsed.depth_max is None
    assert parsed.lat_min is None
    assert parsed.metadata_filters == {}
    
    parsed = parse_query("")
    assert not parsed.has_filters
