"""
router/query_router.py

Intent-based query router for SeaBorg.

Classifies a natural-language question into one of three intents:
  STRUCTURED — answerable via direct PostgreSQL aggregation / row-fetch
  SEMANTIC   — answerable via FAISS vector search + LLM generation
  HYBRID     — contains both structured signals AND semantic signals;
               both retrieval paths will be invoked and their results merged.

Design principles
-----------------
* Deterministic and explainable: pure keyword/regex scoring, no LLM call.
* Two independent signal groups are scored separately.
  - structured_score: stats, depth, geo, range, visualisation signals
  - semantic_score:   explain/describe/why/how/pattern/overview signals
* HYBRID fires when BOTH groups have at least one signal.
* Falls back to SEMANTIC when neither group fires (safe default).
* Every routing decision is logged at INFO level with the matched signals.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


# ── Intent enum ───────────────────────────────────────────────────────────────

class QueryType(Enum):
    STRUCTURED = "structured"
    SEMANTIC   = "semantic"
    HYBRID     = "hybrid"


# ── Routing result dataclass ──────────────────────────────────────────────────

@dataclass
class RoutingResult:
    """Holds the classification decision together with the matched signals."""
    intent:           QueryType
    structured_signals: List[str] = field(default_factory=list)
    semantic_signals:   List[str] = field(default_factory=list)

    @property
    def structured_score(self) -> int:
        return len(self.structured_signals)

    @property
    def semantic_score(self) -> int:
        return len(self.semantic_signals)

    def explain(self) -> str:
        """Human-readable routing explanation."""
        lines = [
            f"Intent: {self.intent.value.upper()}",
            f"  Structured signals ({self.structured_score}): {self.structured_signals or ['none']}",
            f"  Semantic signals   ({self.semantic_score}): {self.semantic_signals or ['none']}",
        ]
        return "\n".join(lines)


# ── Signal definitions ─────────────────────────────────────────────────────────
#
# STRUCTURED SIGNALS
# ------------------
# Each group contributes ONE signal string when it fires, so the score
# cannot be inflated by having many matching words inside the same group.

_STATS_KEYWORDS = [
    "average", "mean", "min", "minimum", "max", "maximum",
    "count", "total", "statistics", "stats", "distribution",
    "median", "standard deviation", "std", "variance",
]

_RANGE_KEYWORDS = [
    "below", "above", "less than", "greater than",
    "deeper than", "shallower than",
    "at least", "at most", "more than", "fewer than",
]

# "between X and Y" — only fires when flanked by numbers (avoids "relationship between")
_BETWEEN_PATTERN = re.compile(r"between\s+\d", re.IGNORECASE)

_GEO_REGIONS = [
    "atlantic ocean", "atlantic",
    "indian ocean",   "indian",
    "pacific ocean",  "pacific",
    "arabian sea",
    "bay of bengal",
    "south china sea",
    "mediterranean sea", "mediterranean",
    "southern ocean",
    "arctic ocean",
    "gulf of mexico",
    "north sea",
    "red sea",
    "caribbean sea",
    "coral sea",
]

_VIZ_KEYWORDS = [
    # Require explicit intent to visualize — drop generic words like "show"/"profile"
    # that also appear in semantic questions.
    "plot", "chart", "graph", "map", "location",
    "visualize", "visualise", "draw",
]

# Depth regex — matches "500m", "500 m", "500 meters", "around 500m" etc.
_DEPTH_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(?:meters?|metres?|m)\b", re.IGNORECASE)

# SEMANTIC SIGNALS
# ----------------
# Exclude "what is" — too generic; "what causes" retained as it's specific.
# Short keywords (why, how, etc.) use word-boundary regex to avoid substring
# false positives (e.g. "how" inside "show", "why" inside "anywhere").
_SEMANTIC_PHRASES = [
    # Multi-word phrases checked first (substring is fine — long enough to be specific)
    "what causes", "what happened", "what can", "tell me about",
    "help me understand",
    # Medium-length keywords (safe as substrings)
    "explain", "describe", "summarize", "summarise", "overview",
    "insight", "insights", "reasoning", "interpret", "interpretation",
    "analysis", "analyze", "analyse", "relationship", "correlation",
]

# Short keywords that must match as whole words to avoid substring false positives
_SEMANTIC_WORD_PATTERNS = [
    re.compile(r"\bwhy\b"),
    re.compile(r"\bhow\b"),
    re.compile(r"\bcause\b"),
    re.compile(r"\bcauses\b"),
    re.compile(r"\breason\b"),
    re.compile(r"\bpattern\b"),
    re.compile(r"\bpatterns\b"),
    re.compile(r"\bimpact\b"),
    re.compile(r"\beffect\b"),
    re.compile(r"\baffects\b"),
    re.compile(r"\binfluence\b"),
]


# ── Core scoring logic ────────────────────────────────────────────────────────

def _score_structured(q: str) -> List[str]:
    """Return list of matched structured signal labels."""
    signals: List[str] = []

    if any(k in q for k in _STATS_KEYWORDS):
        matched = [k for k in _STATS_KEYWORDS if k in q]
        signals.append(f"stats({', '.join(matched)})")

    has_range = any(k in q for k in _RANGE_KEYWORDS) or bool(_BETWEEN_PATTERN.search(q))
    if has_range:
        matched = [k for k in _RANGE_KEYWORDS if k in q]
        if _BETWEEN_PATTERN.search(q):
            matched.append("between <number>")
        signals.append(f"range({', '.join(matched)})")

    if any(k in q for k in _GEO_REGIONS):
        matched = [k for k in _GEO_REGIONS if k in q]
        signals.append(f"geo({', '.join(matched)})")

    if _DEPTH_PATTERN.search(q):
        found = _DEPTH_PATTERN.findall(q)
        signals.append(f"depth_regex(found {len(found)} match(es))")

    if any(k in q for k in _VIZ_KEYWORDS):
        matched = [k for k in _VIZ_KEYWORDS if k in q]
        signals.append(f"viz({', '.join(matched)})")

    return signals


def _score_semantic(q: str) -> List[str]:
    """Return list of matched semantic signal labels (with word-boundary safety)."""
    signals: List[str] = []
    matched: List[str] = []

    # Phrase-level matches (substring OK — long enough to be specific)
    matched.extend(p for p in _SEMANTIC_PHRASES if p in q)

    # Word-boundary matches for short keywords
    matched.extend(
        pat.pattern.replace(r"\b", "")
        for pat in _SEMANTIC_WORD_PATTERNS
        if pat.search(q)
    )

    if matched:
        signals.append(f"semantic({', '.join(matched)})")
    return signals


# ── Public API ────────────────────────────────────────────────────────────────

def route_query(question: str) -> RoutingResult:
    """
    Classify *question* into STRUCTURED, SEMANTIC, or HYBRID.

    Returns a RoutingResult containing the intent and matched signals.
    """
    q = question.lower()

    s_signals = _score_structured(q)
    n_signals = _score_semantic(q)

    has_structured = len(s_signals) > 0
    has_semantic   = len(n_signals) > 0

    if has_structured and has_semantic:
        intent = QueryType.HYBRID
    elif has_structured:
        intent = QueryType.STRUCTURED
    else:
        # Default: SEMANTIC (handles pure semantic AND ambiguous/unknown)
        intent = QueryType.SEMANTIC

    result = RoutingResult(
        intent=intent,
        structured_signals=s_signals,
        semantic_signals=n_signals,
    )

    logger.info("[ROUTER] %s", result.explain())
    return result


def classify_query(question: str) -> QueryType:
    """
    Thin wrapper that returns only the QueryType.

    Preserves the existing public API used by chat.py so no callers break.
    """
    return route_query(question).intent
