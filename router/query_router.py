import re
from enum import Enum


class QueryType(Enum):
    STRUCTURED = "structured"
    SEMANTIC = "semantic"


def classify_query(question: str) -> QueryType:
    q = question.lower()

    # Statistical keywords
    stats_keywords = ["average", "mean", "min", "minimum", "max", "maximum", "count"]
    if any(k in q for k in stats_keywords):
        return QueryType.STRUCTURED

    # Range keywords
    range_keywords = ["below", "above", "less than", "greater than", "between"]
    if any(k in q for k in range_keywords):
        return QueryType.STRUCTURED

    # Geographic keywords
    geo_keywords = [
        "atlantic ocean",
        "indian ocean",
        "pacific ocean",
        "arabian sea",
        "bay of bengal",
        "mediterranean sea",
    ]
    if any(k in q for k in geo_keywords):
        return QueryType.STRUCTURED

    # Depth indicators (e.g. 100m, 200m, 500m, 1000m, \d+m)
    if re.search(r"\d+\s*m\b", q):
        return QueryType.STRUCTURED
    
    visualization_keywords = [
        "profile",
        "plot",
        "chart",
        "graph",
        "map",
        "location",
        "trend",
        "over time"
    ]

    if any(k in q for k in visualization_keywords):
        return QueryType.STRUCTURED

    # Semantic keywords (summarize, explain, describe, trend, pattern, overview, insight)
    # By default, anything else is semantic
    return QueryType.SEMANTIC
