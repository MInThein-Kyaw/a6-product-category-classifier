"""Small keyword-overlap retriever. It does not change A6 model features."""

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent


def retrieve_context(description, limit=2):
    """Return top positive-score records; ties use record ID for repeatability."""
    if not isinstance(description, str):
        return []
    words = set(re.findall(r"[a-z0-9]+", description.lower()))
    records = json.loads((HERE / "knowledge-base.json").read_text(encoding="utf-8"))
    ranked = []
    for record in records:
        keyword_words = set(re.findall(r"[a-z0-9]+", " ".join(record["keywords"]).lower()))
        score = len(words & keyword_words)
        if score:
            ranked.append({"score": score, **record})
    return sorted(ranked, key=lambda record: (-record["score"], record["id"]))[:limit]


def build_decision_request(product_name, description):
    """Application packet: context is metadata, not additional classifier input."""
    return {"input": {"product_name": product_name, "description": description},
            "retrieved_context": retrieve_context(description)}
