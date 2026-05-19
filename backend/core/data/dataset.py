from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Optional

from datasets import Dataset, load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizer

from .template import get_template, PromptTemplate


def _load_raw(path_or_repo: str, format: str) -> List[Dict]:
    p = Path(path_or_repo)
    if p.exists():
        with open(p) as f:
            data = json.load(f) if p.suffix == ".json" else [json.loads(l) for l in f if l.strip()]
        return data
    # HuggingFace Hub dataset
    ds = hf_load_dataset(path_or_repo, split="train")
    return [row for row in ds]


def _alpaca_to_text(row: Dict, template: PromptTemplate) -> str:
    return template.format_example(
        instruction=row.get("instruction", ""),
        input_text=row.get("input", ""),
        output=row.get("output", ""),
    )


def _sharegpt_to_text(row: Dict, template: PromptTemplate) -> str:
    conversations = row.get("conversations") or row.get("messages", [])
    return template.format_messages(conversations)


def _plain_text(row: Dict) -> str:
    return row.get("text", row.get("content", ""))


def build_dataset(
    path_or_repo: str,
    format: str,
    template_name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dataset:
    raw = _load_raw(path_or_repo, format)
    template = get_template(template_name)

    texts = []
    for row in raw:
        if format == "alpaca":
            texts.append(_alpaca_to_text(row, template))
        elif format == "sharegpt":
            texts.append(_sharegpt_to_text(row, template))
        elif format == "plain_text":
            texts.append(_plain_text(row))
        else:
            texts.append(_alpaca_to_text(row, template))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


def build_plain_text_dataset(
    path_or_repo: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dataset:
    """For unsupervised / continued pre-training — raw text only."""
    raw = _load_raw(path_or_repo, "plain_text")
    texts = [_plain_text(r) for r in raw]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds
