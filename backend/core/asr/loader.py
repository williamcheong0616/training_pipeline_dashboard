from __future__ import annotations
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig


def load_whisper_processor(
    model_id: str,
    language: str | None = None,
    task: str = "transcribe",
) -> WhisperProcessor:
    processor = WhisperProcessor.from_pretrained(model_id)
    return processor


def load_whisper_model(
    model_id: str,
    quantization: str | None = None,
    device_map: str = "auto",
    language: str | None = None,
    task: str = "transcribe",
    processor=None,
) -> WhisperForConditionalGeneration:
    kwargs: dict = {
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = WhisperForConditionalGeneration.from_pretrained(model_id, **kwargs)

    if task:
        model.generation_config.task = task
    if language and language != "auto" and processor is not None:
        # Single-language mode: force decoder to the specified language
        model.generation_config.language = language
        model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task=task or "transcribe"
        )
    else:
        # Multilingual / code-mixed mode: let Whisper auto-detect language per segment
        model.generation_config.forced_decoder_ids = None
        model.generation_config.language = None

    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.enable_input_require_grads()
    return model
