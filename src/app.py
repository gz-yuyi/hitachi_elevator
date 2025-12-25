from fastapi import FastAPI, File, Form, UploadFile

from .models import (
    APIResponse,
    DocParseData,
    KnowledgeDeleteRequest,
    KnowledgeFollowItem,
    KnowledgeFollowRequest,
    KnowledgeNeedFollowData,
    KnowledgeNeedFollowRequest,
    KnowledgeSearchRequest,
    KnowledgeUploadRequest,
    SensitiveDeleteRequest,
    SensitiveMatchItem,
    SensitiveMatchRequest,
    SensitiveTrainRequest,
    SmartFillData,
    SmartFillRequest,
    TrappedDetectData,
    TrappedDetectRequest,
)

app = FastAPI(title="日立电梯投标Demo算法需求", version="0.1.0")


@app.post(
    "/hitachi_elevator/smart_fill",
    response_model=APIResponse[SmartFillData],
    summary="智能填单",
    tags=["智能填单"],
)
def smart_fill(_: SmartFillRequest) -> APIResponse[SmartFillData]:
    return APIResponse(data=SmartFillData())


@app.post(
    "/hitachi_elevator/trapped_detect",
    response_model=APIResponse[TrappedDetectData],
    summary="困人困梯检测",
    tags=["困人困梯检测"],
)
def trapped_detect(_: TrappedDetectRequest) -> APIResponse[TrappedDetectData]:
    data = TrappedDetectData(
        is_trapped=False,
        event_type="不确定",
        probability=0.0,
        evidence="",
    )
    return APIResponse(data=data)


@app.post(
    "/hitachi_elevator/sensitive/train",
    response_model=APIResponse[None],
    summary="敏感词训练",
    tags=["敏感词"],
)
def sensitive_train(_: SensitiveTrainRequest) -> APIResponse[None]:
    return APIResponse()


@app.post(
    "/hitachi_elevator/sensitive/delete",
    response_model=APIResponse[None],
    summary="敏感词删除",
    tags=["敏感词"],
)
def sensitive_delete(_: SensitiveDeleteRequest) -> APIResponse[None]:
    return APIResponse()


@app.post(
    "/hitachi_elevator/sensitive/match",
    response_model=APIResponse[list[SensitiveMatchItem]],
    summary="敏感词匹配",
    tags=["敏感词"],
)
def sensitive_match(_: SensitiveMatchRequest) -> APIResponse[list[SensitiveMatchItem]]:
    return APIResponse(data=[])


@app.post(
    "/hitachi_elevator/knowledge/upload",
    response_model=APIResponse[None],
    summary="知识上传",
    tags=["知识库"],
)
def knowledge_upload(_: KnowledgeUploadRequest) -> APIResponse[None]:
    return APIResponse()


@app.post(
    "/hitachi_elevator/knowledge/delete",
    response_model=APIResponse[None],
    summary="知识删除",
    tags=["知识库"],
)
def knowledge_delete(_: KnowledgeDeleteRequest) -> APIResponse[None]:
    return APIResponse()


@app.post(
    "/hitachi_elevator/knowledge/need_follow",
    response_model=APIResponse[KnowledgeNeedFollowData],
    summary="是否需要知识跟随",
    tags=["知识库"],
)
def knowledge_need_follow(
    _: KnowledgeNeedFollowRequest,
) -> APIResponse[KnowledgeNeedFollowData]:
    return APIResponse(data=KnowledgeNeedFollowData(need_follow=False))


@app.post(
    "/hitachi_elevator/knowledge/follow",
    response_model=APIResponse[list[KnowledgeFollowItem]],
    summary="知识跟随",
    tags=["知识库"],
)
def knowledge_follow(
    _: KnowledgeFollowRequest,
) -> APIResponse[list[KnowledgeFollowItem]]:
    return APIResponse(data=[])


@app.post(
    "/hitachi_elevator/knowledge/search",
    response_model=APIResponse[list[KnowledgeFollowItem]],
    summary="知识搜索",
    tags=["知识库"],
)
def knowledge_search(
    _: KnowledgeSearchRequest,
) -> APIResponse[list[KnowledgeFollowItem]]:
    return APIResponse(data=[])


@app.post(
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
