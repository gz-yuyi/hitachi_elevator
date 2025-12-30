import asyncio
import json
from pathlib import Path

import click
import httpx
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


@click.group()
def test_group() -> None:
    """Integration test commands."""


@test_group.command("all")
@click.option("--api-url", default="http://localhost:8000", help="External API URL")
@click.option(
    "--use-test-client/--no-use-test-client",
    default=False,
    help="Use FastAPI TestClient",
)
@click.option("--verbose", is_flag=True, help="Show detailed test output")
def test_all_command(api_url: str, use_test_client: bool, verbose: bool) -> None:
    """Run all integration tests."""
    from click import ClickException
    from src.api.doc_parse import run_integration_tests as test_doc_parse
    from src.api.smart_fill import run_integration_tests as test_smart_fill
    from src.api.trapped_detect import run_integration_tests as test_trapped_detect
    from src.api.knowledge import run_integration_tests as test_knowledge
    from src.api.sensitive import run_integration_tests as test_sensitive

    if verbose:
        click.echo(f"API URL: {api_url}")
        click.echo(f"Using TestClient: {use_test_client}")
        click.echo("=" * 50)

    async def run_all():
        async def run_test(func, name: str):
            try:
                await func(api_url, use_test_client, verbose)
            except ClickException as e:
                raise ClickException(f"{name} failed") from e
            except Exception as e:
                raise ClickException(f"{name} error: {e}") from e

        await run_test(test_doc_parse, "doc_parse")
        await run_test(test_smart_fill, "smart_fill")
        await run_test(test_trapped_detect, "trapped_detect")
        await run_test(test_knowledge, "knowledge")
        await run_test(test_sensitive, "sensitive")

    asyncio.run(run_all())


cli.add_command(start_server)
cli.add_command(export_openapi)
cli.add_command(check_external_service_command)
cli.add_command(test_group)


if __name__ == "__main__":
    cli()
