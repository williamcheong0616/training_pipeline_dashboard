from __future__ import annotations
import re
from typing import Optional
import numpy as np
from datasets import load_dataset, DatasetDict


def build_asr_dataset(
    train_csv: str,
    processor,
    val_csv: Optional[str] = None,
    val_split: float = 0.1,
    sample_rate: int = 16000,
    audio_col: str = "audio_path",
    text_col: str = "text",
    max_label_length: int = 448,
    language: Optional[str] = None,
) -> DatasetDict:
    import librosa

    data_files = {"train": train_csv}
    if val_csv:
        data_files["validation"] = val_csv

    dataset = load_dataset("csv", data_files=data_files)

    if "validation" not in dataset:
        splits = dataset["train"].train_test_split(test_size=val_split, seed=42)
        dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})

    mono_english = language and language.lower() in ("english", "en")

    def prepare_sample(batch):
        text = re.sub(r"<\|.*?\|>", "", str(batch[text_col])).strip()
        if mono_english:
            text = text.lower()
        batch["labels"] = processor.tokenizer(
            text, max_length=max_label_length, truncation=True
        ).input_ids

        audio_path = batch[audio_col]
        try:
            audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
            batch["input_features"] = processor.feature_extractor(
                audio, sampling_rate=sample_rate
            ).input_features[0]
            batch["is_valid"] = True
        except Exception as e:
            print(f"  [WARN] Skipping {audio_path}: {e}")
            batch["input_features"] = np.zeros((80, 3000), dtype=np.float32)
            batch["is_valid"] = False

        return batch

    encoded = dataset.map(
        prepare_sample,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
        writer_batch_size=50,
        desc="Extracting Whisper features",
    )
    encoded = encoded.filter(lambda x: x["is_valid"] is True, num_proc=1)
    encoded = encoded.remove_columns(["is_valid"])
    return encoded
