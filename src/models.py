from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    code: int = Field(default=200, description="业务状态码")
    msg: str = Field(default="success", description="状态说明")
    data: Optional[T] = Field(default=None, description="业务数据")


class ChatTurn(BaseModel):
    role: Literal["user", "agent"] = Field(description="说话角色：user/agent")
    text: str = Field(description="ASR转写文本")


class SmartFillRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "session_id": "call_20241224_001",
                    "call_type": "mid",
                    "turn_id": 12,
                    "history": [
                        {
                            "role": "user",
                            "text": "我们小区电梯这两天老是停在12楼不开门。",
                        },
                        {"role": "agent", "text": "请问具体地址和联系人电话？"},
                    ],
                }
            ]
        }
    )
    session_id: str = Field(description="通话唯一标识")
    call_type: Literal["mid", "final"] = Field(description="mid中间调用，final结束调用")
    turn_id: int = Field(description="当前轮次编号")
    history: list[ChatTurn] = Field(description="对话上下文（时间正序）")


class FillField(BaseModel):
    value: str = Field(default="", description="抽取结果")
    evidence: str = Field(default="", description="命中原文片段")


class SmartFillData(BaseModel):
    complaint_content: FillField = Field(default_factory=FillField, description="咨询/投诉内容")
    address: FillField = Field(default_factory=FillField, description="所在地区地址")
    org_name: FillField = Field(default_factory=FillField, description="使用单位名称")
    contact_name: FillField = Field(default_factory=FillField, description="联系人姓名")
    contact_phone: FillField = Field(default_factory=FillField, description="联系人电话")


class TrappedDetectRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "session_id": "call_20241224_001",
                    "turn_id": 7,
                    "role": "user",
                    "text": "电梯突然停了，我被困在里面了，帮忙联系维修！",
                }
            ]
        }
    )
    session_id: str = Field(description="通话唯一标识")
    turn_id: int = Field(description="当前轮次编号")
    role: Literal["user", "agent"] = Field(description="说话角色：user/agent")
    text: str = Field(description="当前轮次ASR转写文本")


class TrappedDetectData(BaseModel):
    is_trapped: bool = Field(description="是否判定为困人/困梯事件")
    event_type: Literal["困人", "困梯", "不确定"] = Field(description="事件类型")
    probability: float = Field(description="置信度")
    evidence: str = Field(description="触发判断的关键短语")


class SensitiveTrainRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "method": "update",
                    "group_id": "敏感词组ID",
                    "group_name": "敏感词组名称",
                    "group_type": "1",
                    "data": ["投诉", "危害", "敏感词"],
                    "regex": False,
                }
            ]
        }
    )
    method: Literal["update"] = Field(description="固定为 update")
    group_id: str = Field(description="敏感词组ID")
    group_name: str = Field(description="敏感词组名称")
    group_type: str = Field(description="1市民，2话务员")
    data: list[str] = Field(description="敏感词列表")
    regex: bool = Field(default=False, description="是否正则表达式")


class SensitiveDeleteRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"group_id": ["group_id_1", "id"]}]}
    )
    group_id: list[str] = Field(description="要删除的敏感词组ID列表")


class SensitiveMatchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "type": "1",
                    "text": "涉事主体是广州市公安局，地点在广州市白云区",
                    "semantic_score_threshold": 0.99,
                    "semantic_group_threshold": 0.9,
                    "use_rerank_filter_full_match": True,
                }
            ]
        }
    )
    type: str = Field(description="文本类型标识")
    text: str = Field(description="待匹配文本")
    semantic_score_threshold: Optional[float] = Field(
        default=0.99, description="语义候选最低得分"
    )
    semantic_group_threshold: Optional[float] = Field(
        default=0.9, description="同组rerank得分下限"
    )
    use_rerank_filter_full_match: Optional[bool] = Field(
        default=True, description="是否过滤完全匹配结果"
    )


class SensitiveMatchItem(BaseModel):
    group_id: str = Field(description="命中的敏感词组ID")
    group_name: str = Field(description="敏感词组名称")
    probability: float = Field(description="命中置信度")
    matched_terms: list[str] = Field(description="命中的词")


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
    method: Literal["update"] = Field(description="固定为 update")
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
    method: Literal["delete"] = Field(description="固定为 delete")
    data: list[KnowledgeDeleteItem] = Field(description="删除数据列表")


class KnowledgeNeedFollowRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": "我想办理房贷业务"}]})
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


class DocParseData(BaseModel):
    file_name: str = Field(description="文件名")
    file_type: str = Field(description="文件类型")
    page_count: int = Field(description="页数")
    content: str = Field(description="提取文本")
    content_html: Optional[str] = Field(default=None, description="HTML内容")
