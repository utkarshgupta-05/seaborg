import pytest
from structured_query.parser import parse_query

def test_variable_range():
    q = parse_query("salinity above 35")
    assert q.variable == "salinity"
    assert q.value_min == 35.0
    assert q.value_max is None

    q = parse_query("temperature below 10")
    assert q.variable == "temp_c"
    assert q.value_max == 10.0
    assert q.value_min is None

    q = parse_query("chlorophyll between 0.1 and 0.5")
    assert q.variable == "chlorophyll"
    assert q.value_min == 0.1
    assert q.value_max == 0.5
