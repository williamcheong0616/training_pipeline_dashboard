import typer
from backend.cli.train import train_app
from backend.cli.eval import eval_app
from backend.cli.export import export_app

app = typer.Typer(name="llamapipeline", help="LLM Training Pipeline CLI")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")

if __name__ == "__main__":
    app()
