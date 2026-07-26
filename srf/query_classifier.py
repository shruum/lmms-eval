"""
Query classification for adaptive head selection (Strategy 2).

Different query types benefit from different head selections:
- Count queries: Need heads that attend broadly (high head%, broader layers)
- Absence queries: Need selective heads (low head%, focused layers)
- Spatial queries: Need precise localization (medium head%, middle layers)
- Attribute queries: Need detail-focused heads (medium-high head%, later layers)
"""

from __future__ import annotations

import re
from typing import Dict


def classify_query(question: str) -> str:
    """
    Classify query type from question text.

    Args:
        question: The question text (e.g., "Is there a cat in the image?")

    Returns:
        Query type: "count", "absence", "spatial", "attribute", or "other"
    """
    if not question:
        return "other"

    question_lower = question.lower().strip()

    # Count queries: "How many...?", "Count the..."
    count_patterns = [
        r"how many",
        r"count (?:the )?(?:\w+\s*)+",
        r"number of",
        r"total (?:number|count)",
    ]
    if any(re.search(pattern, question_lower) for pattern in count_patterns):
        return "count"

    # Absence queries: "Is there a...?", "Are there any...?"
    absence_patterns = [
        r"is there (?:an?|one|any)",
        r"are there (?:an?|any|some)",
        r"do you see (?:an?|any)",
        r"is this (?:object|item|thing) (?:present|visible|in the image)",
    ]
    if any(re.search(pattern, question_lower) for pattern in absence_patterns):
        return "absence"

    # Spatial queries: "Where is...?", "Which position...?", "At what location...?"
    spatial_patterns = [
        r"where (?:is|are|was|were) (?:the |this |that )?",
        r"which (?:position|location|spot|place|corner|side)",
        r"at what (?:location|position|place|spot)",
        r"(?:left|right|top|bottom|center|middle) (?:side|corner|edge|part)",
    ]
    if any(re.search(pattern, question_lower) for pattern in spatial_patterns):
        return "spatial"

    # Attribute queries: "What color...?", "What size...?", "How does...?"
    attribute_patterns = [
        r"what (?:color|size|shape|texture|pattern|material|style)",
        r"how (?:big|small|large|tall|wide|long|short|thick|thin)",
        r"describe (?:the )?(?:appearance|look|style)",
    ]
    if any(re.search(pattern, question_lower) for pattern in attribute_patterns):
        return "attribute"

    return "other"


def get_head_params(query_type: str) -> Dict[str, float | int]:
    """
    Return head selection parameters for query type.

    Args:
        query_type: One of "count", "absence", "spatial", "attribute", "other"

    Returns:
        Dictionary with head_top_k_pct, layer_start, layer_end
    """
    params = {
        "count": {
            "head_top_k_pct": 0.7,    # Broad: 70% of heads
            "layer_start": 8,          # Earlier start
            "layer_end": 22,           # Later end (broader range)
            "description": "Broad heads for counting distributed objects"
        },
        "absence": {
            "head_top_k_pct": 0.3,    # Selective: 30% of heads
            "layer_start": 12,         # Later start (more refined features)
            "layer_end": 18,           # Narrower range (focused layers)
            "description": "Selective heads for precise absence detection"
        },
        "spatial": {
            "head_top_k_pct": 0.5,    # Balanced: 50% of heads
            "layer_start": 10,         # Middle layers
            "layer_end": 20,           # Wide range for localization
            "description": "Balanced heads for spatial localization"
        },
        "attribute": {
            "head_top_k_pct": 0.6,    # Slightly broad: 60% of heads
            "layer_start": 14,         # Later start (detail-focused)
            "layer_end": 24,           # Latest layers (high-level features)
            "description": "Detail-focused heads for attribute recognition"
        },
        "other": {
            "head_top_k_pct": 0.5,    # Default: 50% of heads
            "layer_start": 10,         # Default range
            "layer_end": 18,
            "description": "Default balanced configuration"
        },
    }

    return params.get(query_type, params["other"])


def get_query_type_description(query_type: str) -> str:
    """Get human-readable description of query type."""
    descriptions = {
        "count": "Count queries (How many? Count the...)",
        "absence": "Absence queries (Is there a...? Are there any...?)",
        "spatial": "Spatial queries (Where is...? Which position...?)",
        "attribute": "Attribute queries (What color...? What size...?)",
        "other": "Other/General queries"
    }
    return descriptions.get(query_type, "Unknown query type")


# Test the classifier with example queries
if __name__ == "__main__":
    test_queries = [
        "Is there a cat in the image?",
        "How many dogs are there?",
        "Where is the car located?",
        "What color is the shirt?",
        "Can you describe the scene?",
    ]

    print("Query Classifier Test:")
    print("=" * 60)
    for q in test_queries:
        q_type = classify_query(q)
        params = get_head_params(q_type)
        print(f"Query: {q}")
        print(f"  Type: {q_type}")
        print(f"  Params: head%={params['head_top_k_pct']}, layers={params['layer_start']}-{params['layer_end']}")
        print(f"  Reasoning: {params['description']}")
        print()