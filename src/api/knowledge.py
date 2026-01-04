from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
import traceback

from ..es import es_client
from ..models import APIResponse
from ..providers import embedding, rerank

router = APIRouter()

KNOWLEDGE_INDEX_PREFIX = "knowledge_"


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


async def create_knowledge_index(group: str):
    index_name = f"{KNOWLEDGE_INDEX_PREFIX}{group}"
    exists = await es_client.client.indices.exists(index=index_name)
    if not exists:
        mapping = {
            "mappings": {
                "properties": {
                    "knowledge_id": {"type": "keyword"},
                    "knowledge_group": {"type": "keyword"},
                    "title": {"type": "text"},
                    "is_message": {"type": "keyword"},
                    "keyword": {"type": "text"},
                    "busi_cata_id": {"type": "keyword"},
                    "busi_cata_name": {"type": "text"},
                    "click_count": {"type": "keyword"},
                    "publish_date": {"type": "date", "format": "yyyy-MM-dd"},
                    "eff_date": {"type": "date", "format": "yyyy-MM-dd"},
                    "content": {"type": "text"},
                    "sys_org_id": {"type": "keyword"},
                    "sys_org_name": {"type": "text"},
                    "update_time": {"type": "date", "format": "yyyy-MM-dd"},
                    "queue_name": {"type": "keyword"},
                    "exp_date": {"type": "date", "format": "yyyy-MM-dd"},
                    "create_time": {"type": "date", "format": "yyyy-MM-dd"},
                    "read_count": {"type": "keyword"},
                    "label": {"type": "keyword"},
                    "knowledge_type_name": {"type": "keyword"},
                    "knowledge_type_path": {"type": "text"},
                    "affair": {"type": "keyword"},
                    "convergence": {"type": "boolean"},
                    "is_bound_follow": {"type": "boolean"},
                    "type_name_label": {"type": "keyword"},
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        await es_client.client.indices.create(index=index_name, body=mapping)


async def knowledge_upload(
    request: KnowledgeUploadRequest,
) -> APIResponse[None]:
    for item in request.data:
        await create_knowledge_index(item.knowledge_group)
        index_name = f"{KNOWLEDGE_INDEX_PREFIX}{item.knowledge_group}"

        content_text = "\n".join(
            [f"{c.sub_title} {c.sub_context}" for c in item.contents if c.sub_context]
        )

        content_vector = await embedding(content_text)

        doc = item.model_dump()
        doc["content"] = content_text
        doc["content_vector"] = content_vector

        await es_client.client.index(
            index=index_name,
            id=item.knowledge_id,
            document=doc,
        )

    return APIResponse()


@router.post(
    "/hitachi_elevator/knowledge/upload",
    response_model=APIResponse[None],
    summary="知识上传",
    tags=["知识库"],
)
async def route_upload(request: KnowledgeUploadRequest) -> APIResponse[None]:
    try:
        return await knowledge_upload(request)
    except Exception as e:
        stack = traceback.format_exc()
        return APIResponse(code=500, msg=f"{e}\n{stack}")


async def knowledge_delete(
    request: KnowledgeDeleteRequest,
) -> APIResponse[None]:
    for item in request.data:
        index_name = f"{KNOWLEDGE_INDEX_PREFIX}{item.knowledge_group}"
        await es_client.client.delete(index=index_name, id=item.knowledge_id)
    return APIResponse()


@router.post(
    "/hitachi_elevator/knowledge/delete",
    response_model=APIResponse[None],
    summary="知识删除",
    tags=["知识库"],
)
async def route_delete(request: KnowledgeDeleteRequest) -> APIResponse[None]:
    try:
        return await knowledge_delete(request)
    except Exception as e:
        stack = traceback.format_exc()
        return APIResponse(code=500, msg=f"{e}\n{stack}")


async def knowledge_need_follow(
    request: KnowledgeNeedFollowRequest,
) -> APIResponse[KnowledgeNeedFollowData]:
    return APIResponse(data=KnowledgeNeedFollowData(need_follow=True))


@router.post(
    "/hitachi_elevator/knowledge/need_follow",
    response_model=APIResponse[KnowledgeNeedFollowData],
    summary="是否需要知识跟随",
    tags=["知识库"],
)
async def route_need_follow(
    request: KnowledgeNeedFollowRequest,
) -> APIResponse[KnowledgeNeedFollowData]:
    try:
        return await knowledge_need_follow(request)
    except Exception as e:
        stack = traceback.format_exc()
        return APIResponse(
            code=500,
            msg=f"{e}\n{stack}",
            data=KnowledgeNeedFollowData(need_follow=False),
        )


async def knowledge_follow(
    request: KnowledgeFollowRequest,
) -> APIResponse[list[KnowledgeFollowItem]]:
    if not request.history:
        return APIResponse(data=[])

    query_text = request.history[-1]
    query_vector = await embedding(query_text)

    index_name = f"{KNOWLEDGE_INDEX_PREFIX}{request.knowledge_group}"

    must_filters = []
    if request.is_bound_follow:
        must_filters.append({"term": {"is_bound_follow": False}})
    if request.affair:
        must_filters.append({"term": {"affair": request.affair}})
    if request.knowledge_type_name:
        type_names = [t.strip() for t in request.knowledge_type_name.split(",")]
        must_filters.append({"terms": {"knowledge_type_name": type_names}})

    knn_query = {
        "field": "content_vector",
        "query_vector": query_vector,
        "k": request.top_k * 2,
        "num_candidates": 100,
    }
    if must_filters:
        knn_query["filter"] = {"bool": {"must": must_filters}}

    search_body = {
        "size": request.top_k * 2,
        "_source": ["*"],
        "knn": knn_query,
    }

    response = await es_client.client.search(index=index_name, body=search_body)

    hits = response["hits"]["hits"]
    if not hits:
        return APIResponse(data=[])

    documents = [hit["_source"] for hit in hits]
    document_texts = [doc.get("content", "") for doc in documents]

    rerank_results = await rerank(document_texts, query_text)

    top_results = []
    for result in rerank_results[: request.top_k]:
        idx = result.index
        if idx < len(documents):
            doc = documents[idx]
            doc["score"] = result.relevance_score
            doc["keywords"] = []
            doc["matched_terms"] = []
            top_results.append(KnowledgeFollowItem(**doc))

    return APIResponse(data=top_results)


@router.post(
    "/hitachi_elevator/knowledge/follow",
    response_model=APIResponse[list[KnowledgeFollowItem]],
    summary="知识跟随",
    tags=["知识库"],
)
async def route_follow(
    request: KnowledgeFollowRequest,
) -> APIResponse[list[KnowledgeFollowItem]]:
    try:
        return await knowledge_follow(request)
    except Exception as e:
        stack = traceback.format_exc()
        return APIResponse(code=500, msg=f"{e}\n{stack}", data=[])


async def knowledge_search(
    request: KnowledgeSearchRequest,
) -> APIResponse[list[KnowledgeFollowItem]]:
    index_name = f"{KNOWLEDGE_INDEX_PREFIX}{request.knowledge_group}"

    must = []
    if request.context_query:
        must.append(
            {
                "query_string": {
                    "fields": ["content", "title", "keyword"],
                    "query": request.context_query,
                }
            }
        )
    if request.sys_org_name:
        must.append({"match": {"sys_org_name": request.sys_org_name}})
    if request.keyword:
        keywords = request.keyword.split()
        must.append(
            {
                "query_string": {
                    "fields": ["content", "title", "keyword"],
                    "query": " ".join(keywords),
                }
            }
        )
    if request.knowledge_type_name:
        must.append({"term": {"knowledge_type_name": request.knowledge_type_name}})
    if request.label:
        must.append({"term": {"label": request.label}})
    if request.is_bound_follow:
        must.append({"term": {"is_bound_follow": False}})

    filter_conditions = []
    if request.queue_name:
        queues = request.queue_name.split()
        filter_conditions.append({"terms": {"queue_name": queues}})
    if request.busi_cata_name:
        cata_names = request.busi_cata_name.split()
        filter_conditions.append({"terms": {"busi_cata_name": cata_names}})
    if request.knowledge_type_path:
        filter_conditions.append(
            {"match": {"knowledge_type_path": request.knowledge_type_path}}
        )

    range_conditions = []
    if request.publish_start and request.publish_end:
        range_conditions.append(
            {
                "range": {
                    "publish_date": {
                        "gte": request.publish_start,
                        "lte": request.publish_end,
                    }
                }
            }
        )

    query = {"bool": {}}
    if must:
        query["bool"]["must"] = must
    if filter_conditions:
        query["bool"]["filter"] = filter_conditions
    if range_conditions:
        if "filter" in query["bool"]:
            query["bool"]["filter"].extend(range_conditions)
        else:
            query["bool"]["filter"] = range_conditions

    search_body = {
        "query": query,
        "from": (request.page_num - 1) * request.page_size,
        "size": request.page_size,
        "_source": ["*"],
    }

    if request.orderby_filed and request.orderby_sort:
        search_body["sort"] = [{request.orderby_filed: {"order": request.orderby_sort}}]

    response = await es_client.client.search(index=index_name, body=search_body)

    hits = response["hits"]["hits"]
    results = []
    for hit in hits:
        doc = hit["_source"]
        doc["score"] = hit["_score"]
        doc["matched_terms"] = []
        results.append(KnowledgeFollowItem(**doc))

    return APIResponse(data=results)


@router.post(
    "/hitachi_elevator/knowledge/search",
    response_model=APIResponse[list[KnowledgeFollowItem]],
    summary="知识搜索",
    tags=["知识库"],
)
async def route_search(
    request: KnowledgeSearchRequest,
) -> APIResponse[list[KnowledgeFollowItem]]:
    try:
        return await knowledge_search(request)
    except Exception as e:
        stack = traceback.format_exc()
        return APIResponse(code=500, msg=f"{e}\n{stack}", data=[])


async def run_integration_tests(
    api_url: str | None = None, use_test_client: bool = False, verbose: bool = False
) -> None:
    """Run knowledge integration tests (minimal sanity checks)."""
    import httpx
    import click

    if verbose:
        click.echo(f"\n{'=' * 50}")
        click.echo("Running Knowledge Integration Tests")
        click.echo(f"API URL: {api_url or 'http://localhost:8000'}")
        click.echo(f"Using TestClient: {use_test_client}")
        click.echo(f"{'=' * 50}\n")

    if use_test_client:
        from ..app import app

        client = httpx.AsyncClient(app=app, base_url="http://test")
    else:
        client = httpx.AsyncClient(base_url=api_url or "http://localhost:8000")

    test_cases = [
        {
            "name": "need follow basic",
            "path": "/hitachi_elevator/knowledge/need_follow",
            "payload": {"text": "我要办证"},
        }
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        try:
            if verbose:
                click.echo(f"\nTest {i}: {test_case['name']}")

            response = await client.post(
                test_case["path"], json=test_case["payload"], timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    passed += 1
                    if verbose:
                        click.echo("  ✓ Passed")
                else:
                    failed += 1
                    if verbose:
                        click.echo(
                            f"  ✗ Failed - Code: {result.get('code')}, Msg: {result.get('msg')}"
                        )
            else:
                failed += 1
                if verbose:
                    click.echo(f"  ✗ Failed - Status: {response.status_code}")
        except Exception as e:
            failed += 1
            if verbose:
                click.echo(f"  ✗ Error - {str(e)}")

    if verbose:
        click.echo(f"\n{'=' * 50}")
        click.echo(f"Tests Passed: {passed}")
        click.echo(f"Tests Failed: {failed}")
        click.echo(f"{'=' * 50}\n")

    if failed > 0:
        raise click.ClickException("")
