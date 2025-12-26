from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from ..models import APIResponse

router = APIRouter()


class KnowledgeContent(BaseModel):
    sub_title: str = Field(default="", description="子标题")
    sub_context: str = Field(default="", description="子标题内容")


class KnowledgeItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    knowledge_id: str = Field(description="知识ID")
    knowledge_group: str = Field(description="知识组")
    title: str = Field(description="知识标题")
    is_message: str = Field(description="短信 0否 1是")
    keyword: str = Field(description="关键词")
    busi_cata_id: str = Field(description="业务目录ID")
    busi_cata_name: str = Field(description="目录名称")
    click_count: str = Field(description="点击次数")
    publish_date: str = Field(description="发布时间")
    eff_date: str = Field(description="生效开始日期")
    sys_org_id: str = Field(description="所属机构ID")
    sys_org_name: str = Field(description="所属机构名称")
    update_time: str = Field(description="更新时间")
    annex_name: str = Field(description="附件名称")
    queue_name: str = Field(description="队列名称")
    exp_date: str = Field(description="失效时间")
    create_time: str = Field(description="创建时间")
    knowledge_type_name: str = Field(description="知识分类名称")
    read_count: str = Field(description="阅读次数")
    label: str = Field(default="", description="标签")
    knowledge_type_path: str = Field(description="知识分类路径")
    affair: list[str] = Field(description="关联事项")
    convergence: bool = Field(default=False, description="是否汇聚性知识")
    is_bound_follow: bool = Field(default=False, description="是否已解绑")
    type_name_label: str = Field(default="", description="知识类型名称标签")
    contents: list[KnowledgeContent] = Field(description="内容结构")


class KnowledgeUploadRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "method": "update",
                    "data": [
                        {
                            "knowledge_id": "id1",
                            "knowledge_group": "group1",
                            "title": "知识点标题",
                            "is_message": "0",
                            "keyword": "关键词",
                            "busi_cata_id": "目录ID",
                            "busi_cata_name": "目录名称",
                            "click_count": "0",
                            "publish_date": "2024-01-01",
                            "eff_date": "2024-01-01",
                            "sys_org_id": "机构ID",
                            "sys_org_name": "机构名称",
                            "update_time": "2024-01-01",
                            "annex_name": "附件名称",
                            "queue_name": "队列名称",
                            "exp_date": "2025-01-01",
                            "create_time": "2024-01-01",
                            "knowledge_type_name": "知识分类名称",
                            "read_count": "0",
                            "label": "标签",
                            "knowledge_type_path": "广州知识库/公安队列/公安业务",
                            "affair": ["关联事项1"],
                            "convergence": False,
                            "is_bound_follow": False,
                            "type_name_label": "政策解读",
                            "contents": [
                                {
                                    "sub_title": "子标题1",
                                    "sub_context": "内容1<br/>",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )
    method: str = Field(description="固定为 update")
    data: list[KnowledgeItem] = Field(description="知识数据列表")


class KnowledgeDeleteItem(BaseModel):
    knowledge_id: str = Field(description="知识ID")
    knowledge_group: str = Field(description="知识组")


class KnowledgeDeleteRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "method": "delete",
                    "data": [{"knowledge_id": "id1", "knowledge_group": "group1"}],
                }
            ]
        }
    )
    method: str = Field(description="固定为 delete")
    data: list[KnowledgeDeleteItem] = Field(description="删除数据列表")


class KnowledgeNeedFollowRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"text": "我想办理房贷业务"}]}
    )
    text: str = Field(description="用户输入文本")


class KnowledgeNeedFollowData(BaseModel):
    need_follow: bool = Field(description="是否需要知识跟随")


class KnowledgeFollowRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "knowledge_group": "group1",
                    "top_k": 4,
                    "affair": "当前关联事项",
                    "knowledge_type_name": "知识分类名称1,知识分类名称2",
                    "is_bound_follow": False,
                    "history": ["我要办证", "办身份证"],
                }
            ]
        }
    )
    knowledge_group: str = Field(description="知识组")
    top_k: int = Field(default=4, description="返回数量")
    affair: str = Field(default="", description="当前关联事项")
    knowledge_type_name: str = Field(default="", description="知识分类名称(逗号分隔)")
    is_bound_follow: bool = Field(default=False, description="是否已解绑")
    history: list[str] = Field(description="对话历史")


class KnowledgeFollowItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    knowledge_id: str = Field(description="知识ID")
    knowledge_group: str = Field(description="知识组")
    title: str = Field(description="知识标题")
    is_message: str = Field(description="短信 0否 1是")
    keyword: str = Field(description="关键词")
    busi_cata_id: str = Field(description="业务目录ID")
    busi_cata_name: str = Field(description="目录名称")
    click_count: str = Field(description="点击次数")
    publish_date: str = Field(description="发布时间")
    eff_date: str = Field(description="生效开始日期")
    content: str = Field(description="知识内容")
    sys_org_id: str = Field(description="所属机构ID")
    sys_org_name: str = Field(description="所属机构名称")
    update_time: str = Field(description="更新时间")
    queue_name: str = Field(description="队列名称")
    exp_date: str = Field(description="失效时间")
    create_time: str = Field(description="创建时间")
    type_name_label: str = Field(default="", description="知识类型名称标签")
    matched_terms: list[str] = Field(description="命中词")
    keywords: list[str] = Field(description="搜索关键词")
    score: float = Field(description="匹配分数")


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "knowledge_group": "group1",
                    "context_query": "搜索内容",
                    "cate_name_query": "类目搜索内容",
                    "sys_org_name": "机构名称",
                    "publish_start": "2024-01-01",
                    "publish_end": "2024-12-31",
                    "queue_name": "队列A",
                    "busi_cata_name": "知识类目",
                    "keyword": "关键字",
                    "annex_search": True,
                    "orderby_filed": "publish_date",
                    "orderby_sort": "desc",
                    "knowledge_type_name": "知识分类名称",
                    "knowledge_type_path": "广州知识库/公安队列/公安业务",
                    "label": "标签",
                    "is_bound_follow": False,
                    "page_size": 15,
                    "page_num": 1,
                }
            ]
        }
    )
    knowledge_group: str = Field(description="知识组")
    context_query: str = Field(default="", description="内容搜索")
    cate_name_query: str = Field(default="", description="类目搜索")
    sys_org_name: str = Field(default="", description="机构名称搜索")
    publish_start: str = Field(default="", description="发布时间开始")
    publish_end: str = Field(default="", description="发布时间结束")
    queue_name: str = Field(default="", description="队列搜索")
    busi_cata_name: str = Field(default="", description="知识类目搜索")
    keyword: str = Field(default="", description="关键字搜索")
    annex_search: bool = Field(default=False, description="是否检索附件")
    orderby_filed: str = Field(default="", description="排序字段")
    orderby_sort: str = Field(default="", description="排序方式")
    knowledge_type_name: str = Field(default="", description="知识分类名称")
    knowledge_type_path: str = Field(default="", description="知识分类路径")
    label: str = Field(default="", description="标签")
    is_bound_follow: bool = Field(default=False, description="是否已解绑")
    page_size: int = Field(default=15, description="每页数量")
    page_num: int = Field(default=10, description="页码")


@router.post(
    "/hitachi_elevator/knowledge/upload",
    response_model=APIResponse[None],
    summary="知识上传",
    tags=["知识库"],
)
def knowledge_upload(_: KnowledgeUploadRequest) -> APIResponse[None]:
    return APIResponse()


@router.post(
    "/hitachi_elevator/knowledge/delete",
    response_model=APIResponse[None],
    summary="知识删除",
    tags=["知识库"],
)
def knowledge_delete(_: KnowledgeDeleteRequest) -> APIResponse[None]:
    return APIResponse()


@router.post(
    "/hitachi_elevator/knowledge/need_follow",
    response_model=APIResponse[KnowledgeNeedFollowData],
    summary="是否需要知识跟随",
    tags=["知识库"],
)
def knowledge_need_follow(
    _: KnowledgeNeedFollowRequest,
) -> APIResponse[KnowledgeNeedFollowData]:
    return APIResponse(data=KnowledgeNeedFollowData(need_follow=False))


@router.post(
    "/hitachi_elevator/knowledge/follow",
    response_model=APIResponse[list[KnowledgeFollowItem]],
    summary="知识跟随",
    tags=["知识库"],
)
def knowledge_follow(
    _: KnowledgeFollowRequest,
) -> APIResponse[list[KnowledgeFollowItem]]:
    return APIResponse(data=[])


@router.post(
    "/hitachi_elevator/knowledge/search",
    response_model=APIResponse[list[KnowledgeFollowItem]],
    summary="知识搜索",
    tags=["知识库"],
)
def knowledge_search(
    _: KnowledgeSearchRequest,
) -> APIResponse[list[KnowledgeFollowItem]]:
    return APIResponse(data=[])
