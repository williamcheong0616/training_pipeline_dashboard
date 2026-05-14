from __future__ import annotations
import typer

export_app = typer.Typer(help="Merge LoRA adapter into base model")


@export_app.command()
def run(
    adapter: str = typer.Option(..., help="Path to LoRA adapter directory"),
    output: str = typer.Option(..., help="Output path for merged model"),
):
    typer.echo(f"Merging {adapter} → {output}")
    from backend.core.model.adapter import merge_and_save
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import json, os

    cfg_path = os.path.join(adapter, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    base = cfg["base_model_name_or_path"]

    typer.echo(f"Loading base model: {base}")
    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter)
    merge_and_save(model, output)

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.save_pretrained(output)
    typer.echo(f"Merged model saved to {output}")
