"""Detect the format of a dataset by scoring sampled records."""
from __future__ import annotations

FORMATS = ["alpaca", "sharegpt", "dpo", "kto", "plain_text"]


def _score_records(records: list[dict]) -> dict[str, int]:
    scores: dict[str, int] = {f: 0 for f in FORMATS}
    for r in records:
        if not isinstance(r, dict):
            continue
        keys = set(r.keys())

        # sharegpt: conversations or messages key containing list-of-dicts
        convs = r.get("conversations") or r.get("messages")
        if isinstance(convs, list) and convs and isinstance(convs[0], dict):
            scores["sharegpt"] += 3

        # dpo: prompt + chosen + rejected
        if {"prompt", "chosen", "rejected"}.issubset(keys):
            scores["dpo"] += 5

        # kto: prompt + completion + label
        if {"prompt", "completion", "label"}.issubset(keys):
            scores["kto"] += 5

        # alpaca: instruction or output present
        if "instruction" in keys or "output" in keys:
            scores["alpaca"] += 2
            if "input" in keys:
                scores["alpaca"] += 1  # full alpaca with context field

        # plain_text: only a text key (and maybe one or two metadata keys)
        if "text" in keys and len(keys) <= 3:
            scores["plain_text"] += 2

    return scores


def detect_format(records: list[dict]) -> dict:
    """
    Score up to 20 records and return the most likely format.

    Returns:
        {
            "format": str,          # best guess
            "confidence": str,      # "high" | "medium" | "low"
            "scores": dict[str,int] # raw scores per format
        }
    """
    if not records:
        return {"format": "plain_text", "confidence": "low", "scores": {f: 0 for f in FORMATS}}

    sample = records[:20]
    scores = _score_records(sample)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    total = sum(scores.values()) or 1
    ratio = scores[best] / total

    confidence = "high" if ratio >= 0.70 else "medium" if ratio >= 0.40 else "low"
    return {"format": best, "confidence": confidence, "scores": scores}
