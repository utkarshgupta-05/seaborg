import inspect
import re
import pytest

from structured_query.engine import answer_structured_query
from structured_query.service import extract_depth
import structured_query.engine as eng_mod
import structured_query.service as svc_mod
import structured_query.repository as repo_mod

@pytest.mark.parametrize("question", [
    "average temperature at 500m",
    "average temperature at 500m in Atlantic Ocean",
    "average salinity at 1000m in Pacific Ocean",
    "temperature between 200m and 400m",
    "deeper than 500m in Indian Ocean",
    "shallower than 100m"
])
def test_structured_query_manual(question):
    """SECTION 1 -- STRUCTURED QUERY MANUAL TESTS"""
    result = answer_structured_query(question)
    assert "summary" in result
    assert "metadata" in result
    assert "rows" in result
    assert "record_count" in result["metadata"]

@pytest.mark.parametrize("question", [
    "average temperature",
    "average salinity",
    "dataset-wide ocean statistics"
])
def test_global_aggregation(question):
    """SECTION 2 -- GLOBAL AGGREGATION TESTS"""
    result = answer_structured_query(question)
    assert "summary" in result
    assert "metadata" in result
    assert "rows" in result

def test_execution_path_verification():
    """SECTION 3 -- EXECUTION PATH VERIFICATION (no Parquet)"""
    eng_src = inspect.getsource(eng_mod)
    svc_src = inspect.getsource(svc_mod)
    repo_src = inspect.getsource(repo_mod)
    
    # Check that read_parquet and load_data are not used
    for mod_name, src in [("engine.py", eng_src), ("service.py", svc_src), ("repository.py", repo_src)]:
        assert "read_parquet" not in src, f"{mod_name} should not use read_parquet"
        assert "load_data" not in src, f"{mod_name} should not use load_data"
        
    # Check SQLAlchemy parameterized query usage
    assert "create_engine" in repo_src or "get_engine" in repo_src
    assert "text(" in repo_src
    assert ":depth_min" in repo_src
    
    # Check for f-string injection vulnerabilities
    user_val_injection = re.search(r'f".*\{depth|f".*\{lat|f".*\{lon', repo_src)
    assert not user_val_injection, f"UNSAFE injection found: {user_val_injection.group() if user_val_injection else ''}"

@pytest.mark.parametrize("phrase", [
    "500m",
    "500 m",
    "500 meters",
    "500 metre",
    "between 200m and 400m",
    "between 200 and 400 meters",
    "deeper than 500m",
    "shallower than 100m",
    "above 100m",
    "below 500m",
    "greater than 500m",
    "less than 100m"
])
def test_depth_parsing_coverage(phrase):
    """SECTION 4 -- DEPTH PARSING COVERAGE"""
    result = extract_depth(phrase)
    assert result != (None, None), f"Failed to extract depth from '{phrase}'"
