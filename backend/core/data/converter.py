"""Convert datasets between supported formats."""
from __future__ import annotations

from .detector import FORMATS

# Which target formats each source format can be converted to
VALID_TARGETS: dict[str, list[str]] = {
    "alpaca":     ["sharegpt", "plain_text"],
    "sharegpt":   ["alpaca",   "plain_text"],
    "dpo":        ["sharegpt", "alpaca"],
    "kto":        ["sharegpt", "alpaca"],
    "plain_text": ["alpaca"],
}

CONVERSION_NOTES: dict[str, dict[str, str]] = {
    "alpaca": {
        "sharegpt":   "Wraps each sample as a two-turn conversation (human/gpt).",
        "plain_text": "Applies the chosen chat template and bakes it into a text field.",
    },
    "sharegpt": {
        "alpaca":     "Extracts the first human/gpt turn as instruction/output.",
        "plain_text": "Applies the chosen chat template and bakes it into a text field.",
    },
    "dpo": {
        "sharegpt": "Keeps chosen responses only; rejected responses are discarded.",
        "alpaca":   "Keeps chosen responses only; rejected responses are discarded.",
    },
    "kto": {
        "sharegpt": "Keeps label=true samples only; negative samples are discarded.",
        "alpaca":   "Keeps label=true samples only; negative samples are discarded.",
    },
    "plain_text": {
        "alpaca": "Wraps the text field as an instruction with an empty output.",
    },
}


# ── per-record converters ────────────────────────────────────────────────────

def _alpaca_to_sharegpt(row: dict) -> dict:
    human = row.get("instruction", "")
    ctx = row.get("input", "")
    if ctx:
        human = f"{human}\n\n{ctx}"
    return {"conversations": [
        {"from": "human", "value": human},
        {"from": "gpt",   "value": row.get("output", "")},
    ]}


def _sharegpt_to_alpaca(row: dict) -> dict:
    convs = row.get("conversations") or row.get("messages", [])
    human = next(
        (c.get("value", "") for c in convs if c.get("from") in ("human", "user")), ""
    )
    gpt = next(
        (c.get("value", "") for c in convs if c.get("from") in ("gpt", "assistant")), ""
    )
    return {"instruction": human, "input": "", "output": gpt}


def _to_plain_text(row: dict, source_fmt: str, template_name: str) -> dict:
    from .template import get_template
    t = get_template(template_name)
    if source_fmt == "alpaca":
        text = t.format_example(
            instruction=row.get("instruction", ""),
            input_text=row.get("input", ""),
            output=row.get("output", ""),
        )
    elif source_fmt == "sharegpt":
        convs = row.get("conversations") or row.get("messages", [])
        text = t.format_messages(convs)
    else:
        text = row.get("text", row.get("content", ""))
    return {"text": text}


def _plain_text_to_alpaca(row: dict) -> dict:
    return {"instruction": row.get("text", row.get("content", "")), "input": "", "output": ""}


# ── normalise dpo / kto to alpaca first ──────────────────────────────────────

def _dpo_to_alpaca(row: dict) -> dict | None:
    """Returns None to signal the row should be dropped (shouldn't happen for dpo)."""
    return {
        "instruction": row.get("prompt", ""),
        "input": "",
        "output": row.get("chosen", ""),
    }


def _kto_to_alpaca(row: dict) -> dict | None:
    if not row.get("label", True):
        return None  # drop negative samples
    return {
        "instruction": row.get("prompt", ""),
        "input": "",
        "output": row.get("completion", ""),
    }


# ── public API ────────────────────────────────────────────────────────────────

def convert_dataset(
    records: list[dict],
    source_fmt: str,
    target_fmt: str,
    template_name: str = "alpaca",
) -> list[dict]:
    """
    Convert a list of records from source_fmt to target_fmt.
    Raises ValueError for unsupported pairings.
    """
    if source_fmt == target_fmt:
        return list(records)

    if target_fmt not in VALID_TARGETS.get(source_fmt, []):
        raise ValueError(
            f"Cannot convert {source_fmt!r} → {target_fmt!r}. "
            f"Valid targets: {VALID_TARGETS.get(source_fmt, [])}"
        )

    out: list[dict] = []

    for row in records:
        # Normalise dpo / kto to alpaca first
        if source_fmt == "dpo":
            row = _dpo_to_alpaca(row)  # type: ignore[assignment]
            effective_src = "alpaca"
        elif source_fmt == "kto":
            row = _kto_to_alpaca(row)
            if row is None:
                continue  # dropped negative sample
            effective_src = "alpaca"
        else:
            effective_src = source_fmt

        # Convert to target
        if target_fmt == "sharegpt":
            if effective_src == "alpaca":
                out.append(_alpaca_to_sharegpt(row))
            else:
                out.append(row)  # already sharegpt
        elif target_fmt == "alpaca":
            if effective_src == "sharegpt":
                out.append(_sharegpt_to_alpaca(row))
            else:
                out.append(row)  # already alpaca
        elif target_fmt == "plain_text":
            out.append(_to_plain_text(row, effective_src, template_name))
        elif target_fmt == "alpaca" and effective_src == "plain_text":
            out.append(_plain_text_to_alpaca(row))

    return out
