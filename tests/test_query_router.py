"""
tests/test_query_router.py

Extensive parametrized tests for the Phase 8 intent-based query router.

Coverage
--------
  Structured  — stats keywords, depth regex, geo regions, range keywords, viz
  Semantic    — explain / describe / why / overview / pattern
  Hybrid      — mixed queries containing both signal groups
  Ambiguous   — edge cases, "meters" vs "m", region without "ocean" suffix
  Edge cases  — empty string, single word, numbers only, garbage text
"""

import pytest

from router.query_router import QueryType, RoutingResult, classify_query, route_query


# ── helpers ───────────────────────────────────────────────────────────────────

def _intent(q: str) -> QueryType:
    return classify_query(q)


def _result(q: str) -> RoutingResult:
    return route_query(q)


# ── STRUCTURED queries ────────────────────────────────────────────────────────

class TestStructuredRouting:
    """Queries that should route to STRUCTURED."""

    @pytest.mark.parametrize("query", [
        "average temperature at 500m",
        "average salinity",
        "average temperature in the dataset",
        "minimum temperature at 200m",
        "count of observations at 1000m",
        "total observations below 300m",
        "temperature statistics at 500m",
        "standard deviation of temperature",
    ])
    def test_stats_keywords_route_structured(self, query):
        assert _intent(query) == QueryType.STRUCTURED

    @pytest.mark.parametrize("query", [
        "maximum salinity in Indian Ocean",   # HYBRID: stats + geo
    ])
    def test_stats_geo_combo_routes_structured_or_hybrid(self, query):
        # geo triggers structured, stats triggers structured — both → STRUCTURED (no semantic)
        assert _intent(query) in (QueryType.STRUCTURED, QueryType.HYBRID)

    @pytest.mark.parametrize("query", [
        "temperature below 300m",
        "salinity above 100m",
        "observations less than 50m deep",
        "readings greater than 1000m",
        "depth between 200m and 400m",
        "deeper than 500m",
        "shallower than 100m",
        "between 200 and 400 meters",
    ])
    def test_range_keywords_route_structured(self, query):
        assert _intent(query) == QueryType.STRUCTURED

    @pytest.mark.parametrize("query", [
        "temperature at 500m in Atlantic Ocean",
        "salinity in the Indian Ocean",
        "observations in Pacific Ocean",
        "profiles in the Arabian Sea",
        "floats in Bay of Bengal",
        "data from Mediterranean Sea",
        "readings in the atlantic",
        "salinity in pacific",
    ])
    def test_geo_regions_route_structured(self, query):
        assert _intent(query) == QueryType.STRUCTURED

    @pytest.mark.parametrize("query", [
        "temperature at 500m",
        "salinity at 1000 m",
        "data at 200 meters",
        "profiles at 300 metres",
        "readings at 750m depth",
        "around 500m depth readings",
    ])
    def test_depth_regex_routes_structured(self, query):
        assert _intent(query) == QueryType.STRUCTURED

    @pytest.mark.parametrize("query", [
        "plot temperature profile and explain what it means",  # viz+semantic
        "show chart of salinity",                              # viz keyword only → STRUCTURED
        "graph depth vs temperature",
        "map float locations",
        "show location of observations",
        "visualize temperature trend over time",
    ])
    def test_visualization_keywords_route_structured_or_hybrid(self, query):
        """Viz keywords always fire a structured signal; if semantic also fires, it's HYBRID."""
        assert _intent(query) in (QueryType.STRUCTURED, QueryType.HYBRID)


# ── SEMANTIC queries ──────────────────────────────────────────────────────────

class TestSemanticRouting:
    """Queries that should route to SEMANTIC."""

    @pytest.mark.parametrize("query", [
        "explain temperature stratification",
        "describe the ocean layers",
        "why does salinity vary with depth",
        "how does pressure affect readings",
        "what causes thermocline formation",
        "summarize float profiles",
        "give me an overview of ocean data",
        "what insights can you find",
        "help me understand the data",
        "what is the impact of depth on oxygen",
        "what patterns exist in the data",
        "what happened to salinity readings",
        "tell me about ARGO float missions",
        "interpret these ocean observations",
    ])
    def test_semantic_keywords_route_semantic(self, query):
        assert _intent(query) == QueryType.SEMANTIC

    @pytest.mark.parametrize("query", [
        # "between" without numbers → no structured signal; "relationship" is semantic
        "analyze the relationship between temperature and salinity",
    ])
    def test_analyze_relationship_between_routes_semantic(self, query):
        """'between' without a number should NOT trigger the range signal."""
        assert _intent(query) == QueryType.SEMANTIC

    @pytest.mark.parametrize("query", [
        "tell me something interesting",
        "what do you know about oceans",
        "",
        "hello",
        "the quick brown fox",
        "123",
        "???",
    ])
    def test_fallback_to_semantic(self, query):
        assert _intent(query) == QueryType.SEMANTIC


# ── HYBRID queries ────────────────────────────────────────────────────────────

class TestHybridRouting:
    """Queries containing both structured and semantic signals → HYBRID."""

    @pytest.mark.parametrize("query", [
        # stats + semantic
        "average temperature at 500m and explain why it varies",
        "what is the mean salinity in Atlantic Ocean and why does it change",
        # depth + semantic
        "temperature at 500m and describe the patterns",
        # geo + semantic
        "salinity in the Indian Ocean — explain what causes these levels",
        "summarize observations in the Pacific Ocean",
        # range + semantic
        "below 300m, describe what the temperature profile looks like",
        "between 200m and 400m, analyze the salinity trends",
        # viz + semantic
        "plot temperature profile and explain what it means",
        "show map of Atlantic and describe the distribution patterns",
        # complex hybrid
        "average temperature deeper than 500m in Indian Ocean — summarize and explain",
        "what is the mean salinity at 1000m and what causes variations",
    ])
    def test_hybrid_queries(self, query):
        assert _intent(query) == QueryType.HYBRID

    def test_hybrid_result_has_both_signal_groups(self):
        result = _result("average temperature at 500m and explain why")
        assert result.intent == QueryType.HYBRID
        assert len(result.structured_signals) > 0
        assert len(result.semantic_signals) > 0
        assert result.structured_score > 0
        assert result.semantic_score > 0


# ── Ambiguous / edge-case queries ─────────────────────────────────────────────

class TestAmbiguousAndEdgeCases:
    """Edge cases the router should handle gracefully."""

    def test_meters_spelled_out(self):
        """'200 meters' should still trigger the depth regex."""
        assert _intent("temperature at 200 meters") == QueryType.STRUCTURED

    def test_metres_alternative_spelling(self):
        """'metres' (British spelling) should still match."""
        assert _intent("depth below 500 metres") == QueryType.STRUCTURED

    def test_region_without_ocean_suffix(self):
        """'atlantic' alone should be a geo signal."""
        assert _intent("data from the atlantic") == QueryType.STRUCTURED

    def test_atlantic_with_explain_is_hybrid(self):
        """'atlantic' + 'explain' should produce HYBRID."""
        assert _intent("explain the data from the atlantic") == QueryType.HYBRID

    def test_stats_word_in_different_position(self):
        """'statistics' anywhere in sentence → STRUCTURED."""
        assert _intent("dataset statistics") == QueryType.STRUCTURED
        # "show" is no longer in viz; but "statistics" alone still fires stats
        assert _intent("can you show me dataset statistics") == QueryType.STRUCTURED

    def test_pure_region_no_explain(self):
        """Just a region name → STRUCTURED (geo signal)."""
        assert _intent("pacific ocean") == QueryType.STRUCTURED

    def test_ocean_mentions_without_data_word(self):
        """Should still route to STRUCTURED on geo match."""
        assert _intent("Indian Ocean salinity") == QueryType.STRUCTURED

    def test_empty_string_falls_back_to_semantic(self):
        assert _intent("") == QueryType.SEMANTIC

    def test_numeric_only_falls_back_to_semantic(self):
        assert _intent("12345") == QueryType.SEMANTIC

    def test_punctuation_only_falls_back_to_semantic(self):
        assert _intent("!!!???...") == QueryType.SEMANTIC

    def test_case_insensitivity(self):
        """Router must be case-insensitive for all signals."""
        assert _intent("AVERAGE TEMPERATURE AT 500M") == QueryType.STRUCTURED
        assert _intent("EXPLAIN THE THERMOCLINE") == QueryType.SEMANTIC
        assert _intent("AVERAGE TEMPERATURE AND EXPLAIN IT") == QueryType.HYBRID

    def test_single_word_explain(self):
        assert _intent("explain") == QueryType.SEMANTIC

    def test_single_word_average(self):
        assert _intent("average") == QueryType.STRUCTURED


# ── RoutingResult dataclass ───────────────────────────────────────────────────

class TestRoutingResult:
    """Unit tests for the RoutingResult dataclass."""

    def test_structured_result_has_no_semantic_signals(self):
        result = _result("average temperature at 500m")
        assert result.intent == QueryType.STRUCTURED
        assert len(result.structured_signals) > 0
        assert len(result.semantic_signals) == 0

    def test_semantic_result_has_no_structured_signals(self):
        result = _result("explain why salinity varies")
        assert result.intent == QueryType.SEMANTIC
        assert len(result.structured_signals) == 0
        assert len(result.semantic_signals) > 0

    def test_explain_method_returns_string(self):
        result = _result("average temperature at 500m")
        explanation = result.explain()
        assert isinstance(explanation, str)
        assert "STRUCTURED" in explanation

    def test_explain_includes_signals(self):
        result = _result("average temperature at 500m in Atlantic Ocean and explain why")
        explanation = result.explain()
        assert "structured_signals" in explanation.lower() or "Structured signals" in explanation
        assert "semantic_signals" in explanation.lower() or "Semantic signals" in explanation

    def test_score_properties(self):
        result = _result("average temperature at 500m and explain why it varies")
        assert result.structured_score == len(result.structured_signals)
        assert result.semantic_score == len(result.semantic_signals)


# ── Backward compatibility ────────────────────────────────────────────────────

class TestBackwardCompatibility:
    """
    Existing callers used classify_query() returning QueryType.
    Verify the old behavior is preserved for all pre-Phase-8 query patterns.
    """

    def test_original_structured_queries_still_work(self):
        legacy_structured = [
            "temperature at 500m",
            "temperature at 500m in Atlantic Ocean",
            "average temperature at 200m",
            "Atlantic Ocean temperature",
            "average salinity",
        ]
        for q in legacy_structured:
            result = _intent(q)
            assert result in (QueryType.STRUCTURED, QueryType.HYBRID), (
                f"Expected STRUCTURED or HYBRID for '{q}', got {result}"
            )

    def test_original_semantic_queries_still_work(self):
        legacy_semantic = [
            "summarize this float profile",
            "describe temperature trends",
            "explain salinity structure",
        ]
        for q in legacy_semantic:
            result = _intent(q)
            assert result in (QueryType.SEMANTIC, QueryType.HYBRID), (
                f"Expected SEMANTIC or HYBRID for '{q}', got {result}"
            )

    def test_classify_query_returns_query_type_enum(self):
        for q in ["average temperature", "explain ocean layers", ""]:
            result = classify_query(q)
            assert isinstance(result, QueryType)
