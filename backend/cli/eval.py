from __future__ import annotations
import typer

eval_app = typer.Typer(help="Evaluate a fine-tuned model")


@eval_app.command()
def run(
    model: str = typer.Option(..., help="Model path"),
    dataset: str = typer.Option(..., help="Evaluation dataset path"),
    batch_size: int = typer.Option(4),
    max_seq_len: int = typer.Option(2048),
):
    from backend.core.model.loader import load_model, load_tokenizer
    from backend.core.data.dataset import build_plain_text_dataset
    import torch, math

    typer.echo(f"Loading model from {model}...")
    tokenizer = load_tokenizer(model)
    model_obj = load_model(model)
    model_obj.eval()

    ds = build_plain_text_dataset(dataset, tokenizer, max_length=max_seq_len)
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)

    total_loss, total_steps = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model_obj.device) for k, v in batch.items()}
            outputs = model_obj(**batch)
            total_loss += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    perplexity = math.exp(avg_loss)
    typer.echo(f"Loss: {avg_loss:.4f}  Perplexity: {perplexity:.2f}")
