from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    code: int = 200
    msg: str = "success"
    data: Optional[T] = None


class ChatTurn(BaseModel):
    role: Literal["user", "agent"]
    text: str


class SmartFillRequest(BaseModel):
    session_id: str
    call_type: Literal["mid", "final"]
    turn_id: int
    history: list[ChatTurn]


class FillField(BaseModel):
    value: str = ""
    evidence: str = ""


class SmartFillData(BaseModel):
    complaint_content: FillField = Field(default_factory=FillField)
    address: FillField = Field(default_factory=FillField)
    org_name: FillField = Field(default_factory=FillField)
    contact_name: FillField = Field(default_factory=FillField)
    contact_phone: FillField = Field(default_factory=FillField)


class TrappedDetectRequest(BaseModel):
    session_id: str
    turn_id: int
    role: Literal["user", "agent"]
    text: str


class TrappedDetectData(BaseModel):
    is_trapped: bool
    event_type: Literal["困人", "困梯", "不确定"]
    probability: float
    evidence: str


class SensitiveTrainRequest(BaseModel):
    method: Literal["update"]
    group_id: str
    group_name: str
    group_type: str
    data: list[str]
    regex: bool = False


class SensitiveDeleteRequest(BaseModel):
    group_id: list[str]


class SensitiveMatchRequest(BaseModel):
    type: str
    text: str
    semantic_score_threshold: Optional[float] = 0.99
    semantic_group_threshold: Optional[float] = 0.9
    use_rerank_filter_full_match: Optional[bool] = True


class SensitiveMatchItem(BaseModel):
    group_id: str
    group_name: str
    probability: float
    matched_terms: list[str]


class KnowledgeContent(BaseModel):
    sub_title: str = ""
    sub_context: str = ""


class KnowledgeItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    knowledge_id: str
    knowledge_group: str
    title: str
    is_message: str
    keyword: str
    busi_cata_id: str
    busi_cata_name: str
    click_count: str
    publish_date: str
    eff_date: str
    sys_org_id: str
    sys_org_name: str
    update_time: str
    annex_name: str
    queue_name: str
    exp_date: str
    create_time: str
    knowledge_type_name: str
    read_count: str
    label: str = ""
    knowledge_type_path: str
    affair: list[str]
    convergence: bool = False
    is_bound_follow: bool = False
    type_name_label: str = ""
    contents: list[KnowledgeContent]


class KnowledgeUploadRequest(BaseModel):
    method: Literal["update"]
    data: list[KnowledgeItem]


class KnowledgeDeleteItem(BaseModel):
    knowledge_id: str
    knowledge_group: str


class KnowledgeDeleteRequest(BaseModel):
    method: Literal["delete"]
    data: list[KnowledgeDeleteItem]


class KnowledgeNeedFollowRequest(BaseModel):
    text: str


class KnowledgeNeedFollowData(BaseModel):
    need_follow: bool


class KnowledgeFollowRequest(BaseModel):
    knowledge_group: str
    top_k: int = 4
    affair: str = ""
    knowledge_type_name: str = ""
    is_bound_follow: bool = False
    history: list[str]


class KnowledgeFollowItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    knowledge_id: str
    knowledge_group: str
    title: str
    is_message: str
    keyword: str
    busi_cata_id: str
    busi_cata_name: str
    click_count: str
    publish_date: str
    eff_date: str
    content: str
    sys_org_id: str
    sys_org_name: str
    update_time: str
    queue_name: str
    exp_date: str
    create_time: str
    type_name_label: str = ""
    matched_terms: list[str]
    keywords: list[str]
    score: float


class KnowledgeSearchRequest(BaseModel):
    knowledge_group: str
    context_query: str = ""
    cate_name_query: str = ""
    sys_org_name: str = ""
    publish_start: str = ""
    publish_end: str = ""
    queue_name: str = ""
    busi_cata_name: str = ""
    keyword: str = ""
    annex_search: bool = False
    orderby_filed: str = ""
    orderby_sort: str = ""
    knowledge_type_name: str = ""
    knowledge_type_path: str = ""
    label: str = ""
    is_bound_follow: bool = False
    page_size: int = 15
    page_num: int = 10


class DocParseData(BaseModel):
    file_name: str
    file_type: str
    page_count: int
    content: str
    content_html: Optional[str] = None
