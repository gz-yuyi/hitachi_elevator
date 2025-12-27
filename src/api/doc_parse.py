import asyncio
import io
import os
import zipfile
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from ..models import APIResponse

router = APIRouter()

MINERU_TOKEN = os.getenv("MINERU_TOKEN")
MINERU_BASE_URL = "https://mineru.net/api/v4"


class DocParseData(BaseModel):
    file_name: str = Field(description="文件名")
    file_type: str = Field(description="文件类型")
    page_count: int = Field(description="页数")
    content: str = Field(description="提取文本")
    content_html: Optional[str] = Field(default=None, description="HTML内容")


def error_response(name: str, file_type: str | None) -> APIResponse[DocParseData]:
    return APIResponse(
        data=DocParseData(
            file_name=name,
            file_type=file_type or "",
            page_count=0,
            content="",
            content_html=None,
        )
    )


async def parse_zip_result(
    zip_url: str, name: str, file_type: str | None, output_fmt: str
) -> APIResponse[DocParseData]:
    async with httpx.AsyncClient(timeout=60) as client:
        zip_resp = await client.get(zip_url)
        if zip_resp.status_code != 200:
            return error_response(name, file_type)

    content_text = ""
    content_html_text = ""
    page_count = 0

    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zip_ref:
        md_files = [f for f in zip_ref.namelist() if f.endswith(".md")]
        html_files = [f for f in zip_ref.namelist() if f.endswith(".html")]

        md_files.sort()
        page_count = len(md_files)

        for i, md_file in enumerate(md_files):
            with zip_ref.open(md_file) as md_f:
                page_content = md_f.read().decode("utf-8")
                if i > 0:
                    content_text += "\n---PAGE_BREAK---\n"
                content_text += page_content

        if output_fmt == "html" and html_files:
            html_files.sort()
            for html_file in html_files:
                with zip_ref.open(html_file) as html_f:
                    content_html_text += html_f.read().decode("utf-8")

    return APIResponse(
        data=DocParseData(
            file_name=name,
            file_type=file_type or name.split(".")[-1].lower(),
            page_count=page_count,
            content=content_text,
            content_html=content_html_text if output_fmt == "html" else None,
        )
    )


async def parse_document(
    file_content: bytes, name: str, file_type: str | None, output_fmt: str
) -> APIResponse[DocParseData]:
    if not MINERU_TOKEN:
        return error_response(name, file_type)

    data_id = name.replace(".", "_").replace(" ", "_")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}",
    }

    upload_req_data = {
        "files": [{"name": name, "data_id": data_id}],
        "model_version": "vlm",
        "extra_formats": ["html"] if output_fmt == "html" else [],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        upload_resp = await client.post(
            f"{MINERU_BASE_URL}/file-urls/batch",
            headers=headers,
            json=upload_req_data,
        )

        if upload_resp.status_code != 200:
            return error_response(name, file_type)

        upload_result_json = upload_resp.json()
        if upload_result_json.get("code") != 0:
            return error_response(name, file_type)

        batch_id = upload_result_json["data"]["batch_id"]
        upload_url = upload_result_json["data"]["file_urls"][0]

        upload_result = await client.put(upload_url, content=file_content)
        if upload_result.status_code != 200:
            return error_response(name, file_type)

    for _ in range(60):
        await asyncio.sleep(2)

        async with httpx.AsyncClient(timeout=30) as client:
            result_resp = await client.get(
                f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}",
                headers=headers,
            )

            if result_resp.status_code != 200:
                continue

            result_json = result_resp.json()
            if result_json.get("code") != 0:
                continue

            result_data = result_json["data"]
            if not result_data.get("extract_result"):
                continue

            extract_result = result_data["extract_result"][0]
            if extract_result["state"] != "done":
                continue

            zip_url = extract_result.get("full_zip_url")
            if not zip_url:
                continue

            return await parse_zip_result(zip_url, name, file_type, output_fmt)

    return error_response(name, file_type)


@router.post(
    "/hitachi_elevator/doc/parse",
    response_model=APIResponse[DocParseData],
    summary="文档解析",
    tags=["文档解析"],
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "example": {
                        "file": "(binary)",
                        "file_name": "电梯维保说明书.pdf",
                        "file_type": "pdf",
                        "output_format": "text",
                    }
                }
            }
        }
    },
)
async def route(
    file: UploadFile = File(..., description="上传文件"),
    file_name: str | None = Form(default=None, description="原始文件名"),
    file_type: str | None = Form(default=None, description="文件类型"),
    output_format: str | None = Form(default=None, description="输出格式：text/html"),
) -> APIResponse[DocParseData]:
    name = file_name or file.filename or ""
    output_fmt = output_format or "text"

    try:
        file_content = await file.read()
        return await parse_document(file_content, name, file_type, output_fmt)
    except Exception as e:
        return error_response(name, file_type)
