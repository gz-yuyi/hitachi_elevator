import asyncio
import json
from pathlib import Path

import click
import uvicorn
import yaml
from dotenv import load_dotenv

from src.app import app
from src.check_external_service import check_external_services

load_dotenv()


@click.group()
def cli() -> None:
    """Hitachi Elevator demo service CLI."""


@click.command("start-server")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--reload/--no-reload", default=False, show_default=True)
def start_server(host: str, port: int, reload: bool) -> None:
    """Start FastAPI server."""
    uvicorn.run("src.app:app", host=host, port=port, reload=reload)


@click.command("export-openapi")
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


@click.command("check-external-service")
def check_external_service_command() -> None:
    """Check if all external services are accessible."""
    asyncio.run(check_external_services())


if __name__ == "__main__":
    cli()
