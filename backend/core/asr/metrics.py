from __future__ import annotations
from typing import Callable
import numpy as np
import evaluate


def make_compute_metrics(processor) -> Callable:
    metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        if len(pred_ids.shape) == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics
