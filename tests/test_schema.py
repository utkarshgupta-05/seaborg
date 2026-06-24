import pytest
from schema.variables import detect_variable

def test_detect_variable_exact_match():
    assert detect_variable("temperature") == "temp_c"
    assert detect_variable("temp") == "temp_c"
    assert detect_variable("salt") == "salinity"
    assert detect_variable("pressure") == "depth_m"

def test_detect_variable_word_boundaries():
    assert detect_variable("what is the temp?") == "temp_c"
    assert detect_variable("give me the temperature's trend") == "temp_c"
    assert detect_variable("saltiness") == "salinity"

def test_detect_variable_no_false_positives():
    # Substring matches should not trigger
    assert detect_variable("template") is None
    assert detect_variable("preserve") is None
    assert detect_variable("depthwise") is None
    assert detect_variable("basalt") is None
    
    # BGC substring false positives
    assert detect_variable("chlorine") is None
    assert detect_variable("machla") is None
    assert detect_variable("dinitrate") is None
    
    # Exact word "thermal" should match
    assert detect_variable("thermal") == "temp_c"

def test_detect_variable_bgc():
    assert detect_variable("chlorophyll levels") == "chlorophyll"
    assert detect_variable("nitrate concentration") == "nitrate"
    assert detect_variable("chl at 50m") == "chlorophyll"
