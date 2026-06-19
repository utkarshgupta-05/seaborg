"""
tests/test_structured_query.py

Unit tests for the PostgreSQL structured query layer (Phase 7).

All database calls are mocked — no live Postgres connection required
for the main suite. A lightweight real-DB integration test is included
at the bottom, guarded by @pytest.mark.skipif.
"""
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_row_df(**kwargs) -> pd.DataFrame:
    """Build a minimal one-row DataFrame that looks like argo_profiles output."""
    defaults = dict(
        float_id="1234567",
        date="2023-06-01",
        latitude=10.0,
        longitude=60.0,
        depth_m=500.0,
        temp_c=12.5,
        salinity=35.2,
        oxygen=200.0,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def _make_stats(**kwargs) -> dict:
    defaults = dict(
        count=42,
        avg_temp=12.5,
        min_temp=10.0,
        max_temp=15.0,
        avg_salinity=35.2,
        min_depth=450.0,
        max_depth=550.0,
    )
    defaults.update(kwargs)
    return defaults


# ── repository tests ──────────────────────────────────────────────────────────

class TestRepository:
    """Tests for structured_query.repository (data-access layer)."""

    def test_no_filters_returns_empty(self):
        """Guard: calling with no filters should return empty, not a full scan."""
        from structured_query.repository import query_with_filters
        result = query_with_filters()
        assert result.empty

    @patch("structured_query.repository._get_engine")
    def test_aggregate_stats_no_filters(self, mock_get_engine):
        """aggregate_stats with no filters should run a global query (no WHERE)."""
        from structured_query.repository import aggregate_stats

        mock_row = MagicMock()
        mock_row._mapping = _make_stats(count=100)

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        mock_get_engine.return_value = mock_engine

        result = aggregate_stats()
        assert result["count"] == 100
        # Check that we executed query without a WHERE clause
        sql_called = str(mock_conn.execute.call_args[0][0])
        assert "WHERE" not in sql_called

    @patch("structured_query.repository._get_engine")
    def test_depth_filter_builds_correct_sql(self, mock_get_engine):
        """Verify the WHERE clause includes depth conditions."""
        from structured_query.repository import query_with_filters

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute = MagicMock()

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        mock_get_engine.return_value = mock_engine

        with patch("structured_query.repository.pd.read_sql", return_value=_make_row_df()) as mock_sql:
            result = query_with_filters(depth_min=450.0, depth_max=550.0)

        assert not result.empty
        call_args = mock_sql.call_args
        sql_str = str(call_args[0][0])  # first positional arg is the text() clause
        assert "depth_m" in sql_str
        assert "LIMIT" in sql_str

    @patch("structured_query.repository._get_engine")
    def test_geo_filter_passes_lat_lon(self, mock_get_engine):
        from structured_query.repository import query_with_filters

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        mock_get_engine.return_value = mock_engine

        with patch("structured_query.repository.pd.read_sql", return_value=_make_row_df()) as mock_sql:
            result = query_with_filters(lat_min=-30.0, lat_max=30.0, lon_min=-70.0, lon_max=20.0)

        assert not result.empty
        sql_str = str(mock_sql.call_args[0][0])
        assert "latitude" in sql_str
        assert "longitude" in sql_str

    @patch("structured_query.repository._get_engine")
    def test_aggregate_stats_returns_dict(self, mock_get_engine):
        from structured_query.repository import aggregate_stats

        mock_row = MagicMock()
        mock_row._mapping = _make_stats()

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        mock_get_engine.return_value = mock_engine

        result = aggregate_stats(depth_min=450.0, depth_max=550.0)
        assert result["count"] == 42
        assert isinstance(result["avg_temp"], float)


# ── service tests ─────────────────────────────────────────────────────────────

class TestService:
    """Tests for structured_query.service (orchestration layer)."""

    @patch("structured_query.service.repository.query_with_filters")
    @patch("structured_query.service.repository.aggregate_stats")
    def test_depth_query_returns_summary(self, mock_stats, mock_query):
        from structured_query.service import answer

        mock_query.return_value = _make_row_df(depth_m=500.0)
        mock_stats.return_value = _make_stats()

        result = answer("temperature at 500m")

        assert "matching observations" in result["summary"]
        assert result["metadata"]["query_type"] == "structured"
        assert result["metadata"]["record_count"] == 1
        assert not result["rows"].empty

    @patch("structured_query.service.repository.query_with_filters")
    @patch("structured_query.service.repository.aggregate_stats")
    def test_region_query_sets_region_metadata(self, mock_stats, mock_query):
        from structured_query.service import answer

        mock_query.return_value = _make_row_df()
        mock_stats.return_value = _make_stats()

        result = answer("average temperature in the Atlantic Ocean")

        assert "region" in result["metadata"]["filters"]
        assert "Atlantic" in result["metadata"]["filters"]["region"]

    @patch("structured_query.service.repository.query_with_filters")
    @patch("structured_query.service.repository.aggregate_stats")
    def test_empty_results_returns_safe_message(self, mock_stats, mock_query):
        from structured_query.service import answer

        mock_query.return_value = pd.DataFrame()
        mock_stats.return_value = {}

        result = answer("temperature at 500m")

        assert result["rows"].empty
        assert result["metadata"]["record_count"] == 0
        assert "No matching" in result["summary"]

    @patch("structured_query.service.repository.aggregate_stats")
    def test_no_filter_question_returns_safe_message(self, mock_stats):
        """A question with no parseable filters returns empty safely if DB has no stats."""
        from structured_query.service import answer
        mock_stats.return_value = {}

        # "hello" has no depth or region
        result = answer("hello")

        assert result["rows"].empty
        assert result["metadata"]["record_count"] == 0
        assert "No matching" in result["summary"]

    @patch("structured_query.service.repository.aggregate_stats")
    def test_no_filter_question_with_global_stats(self, mock_stats):
        """A question with no filters returns global stats but empty rows."""
        from structured_query.service import answer
        mock_stats.return_value = _make_stats(count=150, avg_temp=14.2)

        result = answer("average temperature")

        assert result["rows"].empty
        assert result["metadata"]["record_count"] == 150
        assert "14.20" in result["summary"]
        assert "Found 150 matching observations." in result["summary"]

    @patch("structured_query.service.repository.query_with_filters")
    @patch("structured_query.service.repository.aggregate_stats")
    def test_combined_depth_and_region(self, mock_stats, mock_query):
        from structured_query.service import answer

        mock_query.return_value = _make_row_df()
        mock_stats.return_value = _make_stats()

        result = answer("average temperature at 500m in the Indian Ocean")

        filters = result["metadata"]["filters"]
        assert "depth" in filters
        assert "region" in filters

    @patch("structured_query.service.repository.query_with_filters")
    @patch("structured_query.service.repository.aggregate_stats")
    def test_summary_includes_temp_stats(self, mock_stats, mock_query):
        from structured_query.service import answer

        mock_query.return_value = _make_row_df()
        mock_stats.return_value = _make_stats(avg_temp=13.1, min_temp=11.0, max_temp=15.5)

        result = answer("temperature at 500m")

        assert "13.10" in result["summary"]
        assert "°C" in result["summary"]


# ── engine (public surface) tests ─────────────────────────────────────────────

class TestEngine:
    """Tests for the public answer_structured_query entry point."""

    @patch("structured_query.service.repository.query_with_filters")
    @patch("structured_query.service.repository.aggregate_stats")
    def test_engine_returns_same_shape(self, mock_stats, mock_query):
        from structured_query.engine import answer_structured_query

        mock_query.return_value = _make_row_df()
        mock_stats.return_value = _make_stats()

        result = answer_structured_query("temperature at 500m")

        assert "summary" in result
        assert "rows" in result
        assert "metadata" in result
        assert isinstance(result["rows"], pd.DataFrame)


# ── Depth extraction tests ────────────────────────────────────────────────────

class TestDepthExtraction:
    """Tests for the depth parser."""

    def test_between_range(self):
        from structured_query.parser import parse_query
        p = parse_query("between 200m and 400m")
        assert p.depth_min == 200.0 and p.depth_max == 400.0

    def test_below(self):
        from structured_query.parser import parse_query
        p = parse_query("below 500m")
        assert p.depth_min == 500.0 and p.depth_max is None

    def test_above(self):
        from structured_query.parser import parse_query
        p = parse_query("above 100m")
        assert p.depth_min is None and p.depth_max == 100.0

    def test_at_depth(self):
        from structured_query.parser import parse_query
        p = parse_query("at 500m")
        assert p.depth_min == 450.0
        assert p.depth_max == 550.0

    def test_deeper_than(self):
        from structured_query.parser import parse_query
        p = parse_query("deeper than 300m")
        assert p.depth_min == 300.0 and p.depth_max is None

    def test_shallower_than(self):
        from structured_query.parser import parse_query
        p = parse_query("shallower than 200m")
        assert p.depth_min is None and p.depth_max == 200.0

    def test_no_depth(self):
        from structured_query.parser import parse_query
        p = parse_query("average temperature")
        assert p.depth_min is None and p.depth_max is None

    def test_meters_unit(self):
        from structured_query.parser import parse_query
        p = parse_query("at 500 meters")
        assert p.depth_min == 450.0
        assert p.depth_max == 550.0


# ── Real-DB integration test (guarded) ───────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")

@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set — skipping real-DB test")
class TestRepositoryIntegration:
    """
    Lightweight integration tests against a real Postgres instance.
    These are skipped in CI unless DATABASE_URL is set.
    They are never a dependency for the unit test suite.
    """

    def test_query_returns_dataframe(self):
        from structured_query.repository import query_with_filters
        # A very wide depth range should return something if data exists
        df = query_with_filters(depth_min=0.0, depth_max=5000.0, limit=5)
        assert isinstance(df, pd.DataFrame)

    def test_aggregate_stats_returns_dict(self):
        from structured_query.repository import aggregate_stats
        stats = aggregate_stats(depth_min=0.0, depth_max=5000.0)
        assert isinstance(stats, dict)
        if stats:
            assert "count" in stats
            assert "avg_temp" in stats

    def test_global_aggregate_stats(self):
        from structured_query.repository import aggregate_stats
        stats = aggregate_stats()
        assert isinstance(stats, dict)
        if stats:
            assert "count" in stats
