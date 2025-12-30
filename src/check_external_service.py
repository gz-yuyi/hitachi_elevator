import asyncio
import os

import click
import httpx
from openai import AsyncOpenAI, OpenAIError


async def check_elasticsearch() -> tuple[bool, str]:
    es_host = os.getenv("ES_HOST", "localhost:9200")
    es_username = os.getenv("ES_USERNAME", "")
    es_password = os.getenv("ES_PASSWORD", "")

    try:
        auth = None
        if es_username and es_password:
            auth = (es_username, es_password)

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"http://{es_host}",
                auth=auth,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200:
                info = response.json()
                cluster_name = info.get("cluster_name", "unknown")
                return True, f"Connected (Cluster: {cluster_name})"
            else:
                return False, f"Failed with status {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


async def check_openai() -> tuple[bool, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        return False, "OPENAI_API_KEY not configured"
    if not base_url:
        return False, "OPENAI_BASE_URL not configured"
    if not model:
        return False, "OPENAI_MODEL not configured"

    base_url = base_url.rstrip("/")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30)

    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}]
        )
        return True, f"Connected (Model: {model})"
    except OpenAIError as e:
        status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
        suffix = f" {status}" if status else ""
        return False, f"Failed{suffix}: {str(e)}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


async def check_mineru() -> tuple[bool, str]:
    token = os.getenv("MINERU_TOKEN")
    base_url = "https://mineru.net/api/v4"

    if not token:
        return False, "Token not configured"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
            response = await client.get(
                f"{base_url}/extract-results/batch/test123", headers=headers
            )
            if response.status_code in [200, 400]:
                return True, "Connected (API accessible)"
            else:
                return False, f"Failed with status {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


async def check_xinference() -> tuple[bool, str]:
    host = os.getenv("XINFERENCE_HOST")

    if not host:
        return False, "Host not configured"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"http://{host}/v1/models")
            if response.status_code == 200:
                return True, "Connected"
            else:
                return False, f"Failed with status {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


async def check_tei() -> tuple[bool, str]:
    embedding_url = os.getenv("TIE_EMBEDDING_URL", "")
    rerank_url = os.getenv("TIE_RERANK_URL", "")
    api_key = os.getenv("TEI_API_KEY", "")

    if not embedding_url:
        return False, "Embedding URL not configured"

    results = []
    for url_name, url in [("Embedding", embedding_url), ("Rerank", rerank_url)]:
        if not url:
            continue
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    results.append(f"{url_name}: OK")
                else:
                    results.append(f"{url_name}: Failed ({response.status_code})")
        except Exception as e:
            results.append(f"{url_name}: Error ({str(e)})")

    if results:
        return True, ", ".join(results)
    return False, "No URLs configured"


async def check_vllm() -> tuple[bool, str]:
    embedding_url = os.getenv("VLLM_URL", "")
    rerank_url = os.getenv("VLLM_RERANK_URL", "")
    api_key = os.getenv("VLLM_API_KEY", "")

    if not embedding_url:
        return False, "Embedding URL not configured"

    results = []
    for url_name, url in [("Embedding", embedding_url), ("Rerank", rerank_url)]:
        if not url:
            continue
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    url + "/health" if not url.endswith("/health") else url,
                    headers=headers,
                )
                if response.status_code == 200:
                    results.append(f"{url_name}: OK")
                else:
                    results.append(f"{url_name}: Failed ({response.status_code})")
        except Exception as e:
            results.append(f"{url_name}: Error ({str(e)})")

    if results:
        return True, ", ".join(results)
    return False, "No URLs configured"


async def check_siliconflow() -> tuple[bool, str]:
    token = os.getenv("SILICONFLOW_TOKEN")
    base_url = "https://api.siliconflow.cn"

    if not token:
        return False, "Token not configured"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{base_url}/v1/user/info", headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return True, "Connected"
            else:
                return False, f"Failed with status {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


async def check_external_services() -> None:
    """Check if all external services are accessible."""
    click.echo("Checking external services connectivity...\n")

    checks = [
        ("Elasticsearch", check_elasticsearch()),
        ("OpenAI", check_openai()),
        ("MinerU", check_mineru()),
    ]

    backend = os.getenv("SMALL_MODEL_BACKEND")
    if backend == "xinference":
        checks.append(("Xinference", check_xinference()))
    elif backend == "tei":
        checks.append(("TEI", check_tei()))
    elif backend == "vllm":
        checks.append(("VLLM", check_vllm()))
    elif backend == "siliconflow":
        checks.append(("SiliconFlow", check_siliconflow()))
    else:

        async def check_backend() -> tuple[bool, str]:
            return False, "Not configured"

        checks.append(("Small Model Backend", check_backend()))

    for name, coro in checks:
        success, message = await coro
        status_icon = (
            click.style("✓", fg="green") if success else click.style("✗", fg="red")
        )
        click.echo(f"{status_icon} {name:20s} - {message}")

    click.echo("\nConfiguration Summary:")
    click.echo(f"  Small Model Backend: {backend or 'Not configured'}")
