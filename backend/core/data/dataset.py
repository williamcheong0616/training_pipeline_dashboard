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
    """Return a Dataset with a 'text' column for SFTTrainer to tokenize internally.

    Returning raw text (not pre-tokenized) avoids the shape mismatch that occurs when
    SFTTrainer's processing_class looks for a 'text' column and finds none, producing
    0-length input_ids.
    """
    raw = _load_raw(path_or_repo, format)
    template = get_template(template_name)

    texts = []
    for row in raw:
        if format == "alpaca":
            text = _alpaca_to_text(row, template)
        elif format == "sharegpt":
            text = _sharegpt_to_text(row, template)
        elif format == "plain_text":
            text = _plain_text(row)
        else:
            text = _alpaca_to_text(row, template)
        if text and text.strip():
            texts.append(text)

    if not texts:
        raise ValueError(
            f"Dataset at {path_or_repo!r} produced 0 non-empty samples after formatting."
        )

    # Tell SFTTrainer where to truncate — it reads model_max_length from the tokenizer.
    tokenizer.model_max_length = max_length

    return Dataset.from_dict({"text": texts})


def build_plain_text_dataset(
    path_or_repo: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dataset:
    """For unsupervised / continued pre-training — returns a tokenized dataset.

    Used with plain Trainer + DataCollatorForLanguageModeling which expects input_ids.
    """
    raw = _load_raw(path_or_repo, "plain_text")
    texts = [t for r in raw if (t := _plain_text(r)) and t.strip()]

    if not texts:
        raise ValueError(
            f"Dataset at {path_or_repo!r} produced 0 non-empty plain-text samples."
        )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    return ds
