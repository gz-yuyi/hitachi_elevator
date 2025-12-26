from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from ..models import APIResponse

router = APIRouter()


class DocParseData(BaseModel):
    file_name: str = Field(description="文件名")
    file_type: str = Field(description="文件类型")
    page_count: int = Field(description="页数")
    content: str = Field(description="提取文本")
    content_html: Optional[str] = Field(default=None, description="HTML内容")


def doc_parse(
    file: UploadFile = File(..., description="上传文件"),
    file_name: str | None = Form(default=None, description="原始文件名"),
    file_type: str | None = Form(default=None, description="文件类型"),
    output_format: str | None = Form(default=None, description="输出格式：text/html"),
) -> APIResponse[DocParseData]:
    name = file_name or file.filename or ""
    data = DocParseData(
        file_name=name,
        file_type=file_type or "",
        page_count=0,
        content="",
        content_html="" if output_format == "html" else None,
    )
    return APIResponse(data=data)


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
def route(
    file: UploadFile = File(..., description="上传文件"),
    file_name: str | None = Form(default=None, description="原始文件名"),
    file_type: str | None = Form(default=None, description="文件类型"),
    output_format: str | None = Form(default=None, description="输出格式：text/html"),
) -> APIResponse[DocParseData]:
    return doc_parse(file, file_name, file_type, output_format)
