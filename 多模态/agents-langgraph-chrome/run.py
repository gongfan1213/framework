import os
import typer
from rich import print
from .graph import build_graph


app = typer.Typer()


@app.command()
def main(url: str, goal: str):
    os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9000/v1")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    graph = build_graph()
    state = {"url": url, "goal": goal}
    out = graph.invoke(state)
    print({
        "plan": out.get("plan"),
        "logs": out.get("logs"),
        "answer": out.get("answer")
    })


if __name__ == "__main__":
    app()


