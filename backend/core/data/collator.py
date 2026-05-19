from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, PreTrainedTokenizer


@dataclass
class SFTDataCollator(DataCollatorForSeq2Seq):
    """Wraps DataCollatorForSeq2Seq with label masking for instruction tokens."""
    pass


def get_clm_collator(tokenizer: PreTrainedTokenizer, mlm: bool = False):
    """Collator for causal language modeling (unsupervised / CPT)."""
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)
