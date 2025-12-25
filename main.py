import json
from pathlib import Path

import click
import uvicorn
import yaml

from src.app import app


@click.group()
def cli() -> None:
    """Hitachi Elevator demo service CLI."""


@cli.command("start-server")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--reload/--no-reload", default=False, show_default=True)
def start_server(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server."""
    uvicorn.run("src.app:app", host=host, port=port, reload=reload)


@cli.command("export-openapi")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml"]),
    default="json",
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
def export_openapi(output_format: str, output: Path | None) -> None:
    """Export OpenAPI schema for API tools."""
    schema = app.openapi()
    if output_format == "yaml":
        content = yaml.safe_dump(schema, allow_unicode=True, sort_keys=False)
    else:
        content = json.dumps(schema, ensure_ascii=False, indent=2)

    if output:
        output.write_text(content, encoding="utf-8")
        click.echo(f"Wrote schema to {output}")
    else:
        click.echo(content)


if __name__ == "__main__":
    cli()
