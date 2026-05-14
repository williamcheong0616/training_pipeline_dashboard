from __future__ import annotations
from typing import Optional
import typer

train_app = typer.Typer(help="Run a training job locally")


@train_app.command()
def run(
    model: str = typer.Option(..., help="HF model repo or local path"),
    method: str = typer.Option("sft", help="Training method: sft|unsupervised|dpo|rm|kto|orpo"),
    dataset: str = typer.Option(..., help="Dataset path or HF dataset name"),
    dataset_format: str = typer.Option("alpaca", help="Dataset format: alpaca|sharegpt|plain_text"),
    template: str = typer.Option("alpaca", help="Prompt template name"),
    peft: str = typer.Option("lora", help="PEFT method: lora|qlora|dora|full"),
    quantization: Optional[str] = typer.Option(None, help="Quantization: 4bit|8bit"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Per-device batch size"),
    lr: float = typer.Option(2e-4, help="Learning rate"),
    max_seq_len: int = typer.Option(2048, help="Maximum sequence length"),
    output_dir: str = typer.Option("./outputs/run", help="Output directory"),
):
    config = {
        "model_path": model,
        "dataset_path": dataset,
        "dataset_format": dataset_format,
        "template": template,
        "peft_method": peft,
        "quantization": quantization,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_seq_length": max_seq_len,
        "output_dir": output_dir,
    }

    typer.echo(f"Starting {method.upper()} training → {output_dir}")

    from backend.workers.training_worker import _get_trainer
    trainer = _get_trainer(method, job_id=0, config=config, db_factory=None)
    trainer.train()
    typer.echo(f"Training complete. Model saved to {output_dir}")
