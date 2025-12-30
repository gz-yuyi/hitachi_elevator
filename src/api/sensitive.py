import asyncio
import os
import re
from datetime import datetime
from typing import Any, ClassVar, Optional

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from ..models import APIResponse as APIResponse
from ..providers import embedding, rerank

# Elasticsearch配置
ES_HOST = os.getenv("ES_HOST", "localhost:9200")
if not ES_HOST.startswith(("http://", "https://")):
    ES_HOST = f"http://{ES_HOST}"
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "changeme")

# 索引名称
SENSITIVE_WORDS_INDEX = "sensitive_words"
CLASSIFICATION_INDEX = "text_classification"


class ESClient:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> AsyncElasticsearch:
        if self._client is None:
            self._client = AsyncElasticsearch(
                hosts=[ES_HOST],
                basic_auth=(ES_USERNAME, ES_PASSWORD),
                verify_certs=False,
                ssl_show_warn=False,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None


# 全局ES客户端实例
es_client = ESClient()


async def ensure_index_exists():
    """确保敏感词索引存在"""
    client = es_client.client

    # 检查索引是否存在
    if not await client.indices.exists(index=SENSITIVE_WORDS_INDEX, ignore=[400]):
        # 创建索引映射
        mapping = {
            "mappings": {
                "properties": {
                    "group_id": {"type": "keyword"},
                    "group_name": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "group_type": {"type": "keyword"},
                    "word": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "original_word": {"type": "keyword"},  # 存储原始词汇用于完全匹配
                    "is_regex": {"type": "boolean"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1024,  # BAAI/bge-m3的向量维度
                        "similarity": "cosine",
                    },
                    "created_at": {"type": "date"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
        }

        await client.indices.create(index=SENSITIVE_WORDS_INDEX, body=mapping)
        logger.info(f"Created index: {SENSITIVE_WORDS_INDEX}")


async def index_sensitive_words(
    group_id: str,
    group_name: str,
    group_type: str,
    words: list[str],
    embeddings: list[list[float]],
    is_regex: bool = False,
    delete_old: bool = True,
):
    """索引敏感词数据"""
    client = es_client.client

    # 确保索引存在
    await ensure_index_exists()

    # 如果需要删除旧数据
    if delete_old:
        await delete_by_group_id(group_id)

    # 准备批量索引的文档
    docs = []
    current_time = datetime.now().isoformat()

    for word, embedding in zip(words, embeddings, strict=False):
        # 验证embedding维度
        if len(embedding) != 1024:
            logger.warning(
                f"Embedding dimension mismatch for word '{word}': expected 1024, got {len(embedding)}"
            )
            continue

        doc = {
            "_index": SENSITIVE_WORDS_INDEX,
            "_source": {
                "group_id": group_id,
                "group_name": group_name,
                "group_type": group_type,
                "word": word,
                "original_word": word,
                "is_regex": is_regex,
                "embedding": embedding,
                "created_at": current_time,
            },
        }
        docs.append(doc)

    # 批量索引
    if docs:
        success_count, failed_items = await async_bulk(client, docs, stats_only=False)
        if failed_items:
            logger.error(
                f"Failed to index {len(failed_items)} documents: {failed_items}"
            )
        await client.indices.refresh(index=SENSITIVE_WORDS_INDEX)
        logger.info(
            f"Successfully indexed {success_count} documents for group {group_id}"
        )


async def delete_by_group_id(group_id: str):
    """根据group_id删除文档"""
    client = es_client.client

    query = {"query": {"term": {"group_id": group_id}}}

    await client.delete_by_query(index=SENSITIVE_WORDS_INDEX, body=query)
    logger.info(f"Deleted documents for group {group_id}")


async def delete_by_group_ids(group_ids: list[str]):
    """根据group_ids批量删除文档"""
    client = es_client.client

    query = {"query": {"terms": {"group_id": group_ids}}}

    result = await client.delete_by_query(index=SENSITIVE_WORDS_INDEX, body=query)
    logger.info(f"Deleted {result['deleted']} documents for groups {group_ids}")


async def exact_match_search(text: str, group_type: str) -> list[dict[str, Any]]:
    """完全匹配搜索"""
    client = es_client.client

    # 构建查询条件
    must_conditions = []

    # 只有当 group_type 不为空字符串时才添加类型过滤
    if group_type.strip():
        must_conditions.append({"term": {"group_type": group_type}})

    must_conditions.append(
        {
            "bool": {
                "should": [
                    # 精确匹配原始词汇
                    {"term": {"original_word": text}},
                    # 或者文本中包含敏感词
                    {"match": {"word": text}},
                ]
            }
        }
    )

    # 构建查询
    query = {
        "query": {"bool": {"must": must_conditions}},
        "size": 1000,
    }

    response = await client.search(index=SENSITIVE_WORDS_INDEX, body=query)
    hits = response["hits"]["hits"]

    results = []
    for hit in hits:
        source = hit["_source"]

        # 检查是否真正匹配
        matched = False
        original_word = source["original_word"]

        if source["is_regex"]:
            # 正则表达式匹配
            try:
                if re.search(original_word, text):
                    matched = True
            except re.error:
                logger.warning(f"Invalid regex pattern: {original_word}")
                continue
        else:
            # 完全匹配或包含匹配
            if original_word in text:
                matched = True

        if matched:
            results.append(
                {
                    "group_id": source["group_id"],
                    "group_name": source["group_name"],
                    "probability": 1.0,  # 完全匹配的分值为1
                    "matched_terms": [original_word],
                    "_score": hit["_score"],
                }
            )

    return results


async def vector_search(
    text: str, text_embedding: list[float], group_type: str, k: int = 100
) -> list[dict[str, Any]]:
    """向量相似度搜索"""
    client = es_client.client

    knn_query = {
        "field": "embedding",
        "query_vector": text_embedding,
        "k": k,
        "num_candidates": k * 2,  # 通常设置为 k 的 2-10 倍
    }

    # 只有当 group_type 不为空字符串时才添加类型过滤
    if group_type.strip():
        knn_query["filter"] = {"term": {"group_type": group_type}}

    query = {"knn": knn_query}

    response = await client.search(index=SENSITIVE_WORDS_INDEX, body=query)
    hits = response["hits"]["hits"]

    results = []
    for hit in hits:
        source = hit["_source"]
        # KNN 搜索的分值直接来自相似度计算
        similarity_score = hit["_score"]

        results.append(
            {
                "group_id": source["group_id"],
                "group_name": source["group_name"],
                "original_word": source["original_word"],
                "similarity_score": similarity_score,
                "_score": hit["_score"],
            }
        )

    return results


async def get_all_groups(group_type: str) -> list[dict[str, Any]]:
    """获取所有敏感词组信息"""
    client = es_client.client

    query = {
        "query": {"term": {"group_type": group_type}},
        "aggs": {
            "groups": {
                "terms": {"field": "group_id", "size": 1000},
                "aggs": {
                    "group_info": {
                        "top_hits": {"size": 1, "_source": ["group_name", "group_type"]}
                    },
                    "words": {"terms": {"field": "original_word", "size": 1000}},
                },
            }
        },
        "size": 0,
    }

    response = await client.search(index=SENSITIVE_WORDS_INDEX, body=query)

    groups = []
    for bucket in response["aggregations"]["groups"]["buckets"]:
        group_id = bucket["key"]
        group_info = bucket["group_info"]["hits"]["hits"][0]["_source"]
        words = [word_bucket["key"] for word_bucket in bucket["words"]["buckets"]]

        groups.append(
            {
                "group_id": group_id,
                "group_name": group_info["group_name"],
                "group_type": group_info["group_type"],
                "words": words,
            }
        )

    return groups


async def ensure_classification_index_exists():
    """确保分类索引存在"""
    client = es_client.client

    # 检查索引是否存在
    if not await client.indices.exists(index=CLASSIFICATION_INDEX):
        # 创建索引映射
        mapping = {
            "mappings": {
                "properties": {
                    "group_id": {"type": "keyword"},
                    "group_name": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "group_type": {"type": "keyword"},
                    "intent_id": {"type": "keyword"},
                    "intent_name": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "intent_type": {"type": "keyword"},
                    "example": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "original_example": {"type": "keyword"},  # 存储原始例子用于完全匹配
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1024,  # BAAI/bge-m3的向量维度
                        "similarity": "cosine",
                    },
                    "created_at": {"type": "date"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
        }

        await client.indices.create(index=CLASSIFICATION_INDEX, body=mapping)
        logger.info(f"Created index: {CLASSIFICATION_INDEX}")


async def index_classification_data(
    group_id: str,
    group_name: str,
    group_type: str,
    intent_id: str,
    intent_name: str,
    intent_type: str,
    examples: list[str],
    embeddings: list[list[float]],
    delete_old: bool = True,
):
    """索引分类训练数据"""
    client = es_client.client

    # 确保索引存在
    await ensure_classification_index_exists()

    # 如果需要删除旧数据
    if delete_old:
        await delete_classification_by_intent_id(group_id, intent_id)

    # 准备批量索引的文档
    docs = []
    current_time = datetime.now().isoformat()

    for example, embedding in zip(examples, embeddings, strict=False):
        # 验证embedding维度
        if len(embedding) != 1024:
            logger.warning(
                f"Embedding dimension mismatch for example '{example}': expected 1024, got {len(embedding)}"
            )
            continue

        doc = {
            "_index": CLASSIFICATION_INDEX,
            "_source": {
                "group_id": group_id,
                "group_name": group_name,
                "group_type": group_type,
                "intent_id": intent_id,
                "intent_name": intent_name,
                "intent_type": intent_type,
                "example": example,
                "original_example": example,
                "embedding": embedding,
                "created_at": current_time,
            },
        }
        docs.append(doc)

    # 批量索引
    if docs:
        success_count, failed_items = await async_bulk(client, docs, stats_only=False)
        if failed_items:
            logger.error(
                f"Failed to index {len(failed_items)} documents: {failed_items}"
            )
        await client.indices.refresh(index=CLASSIFICATION_INDEX)
        logger.info(
            f"Successfully indexed {success_count} documents for intent {intent_id} in group {group_id}"
        )


async def delete_classification_by_intent_id(group_id: str, intent_id: str):
    """根据group_id和intent_id删除文档"""
    client = es_client.client

    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"group_id": group_id}},
                    {"term": {"intent_id": intent_id}},
                ]
            }
        }
    }

    await client.delete_by_query(index=CLASSIFICATION_INDEX, body=query)
    logger.info(f"Deleted documents for intent {intent_id} in group {group_id}")


async def delete_classification_by_intent_ids(group_id: str, intent_ids: list[str]):
    """根据group_id和intent_ids批量删除文档"""
    client = es_client.client

    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"group_id": group_id}},
                    {"terms": {"intent_id": intent_ids}},
                ]
            }
        }
    }

    result = await client.delete_by_query(index=CLASSIFICATION_INDEX, body=query)
    logger.info(
        f"Deleted {result['deleted']} documents for intents {intent_ids} in group {group_id}"
    )


async def delete_classification_by_group_id(group_id: str):
    """根据group_id删除所有文档"""
    client = es_client.client

    query = {"query": {"term": {"group_id": group_id}}}

    result = await client.delete_by_query(index=CLASSIFICATION_INDEX, body=query)
    logger.info(f"Deleted {result['deleted']} documents for group {group_id}")


async def exact_match_classification_search(
    text: str, group_id: str
) -> list[dict[str, Any]]:
    """分类的完全匹配搜索"""
    client = es_client.client

    # 构建查询
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"group_id": group_id}},
                    {
                        "bool": {
                            "should": [
                                # 精确匹配原始例子
                                {"term": {"original_example": text}},
                                # 或者文本中包含例子
                                {"match": {"example": text}},
                            ]
                        }
                    },
                ]
            }
        },
        "size": 1000,
    }

    response = await client.search(index=CLASSIFICATION_INDEX, body=query)
    hits = response["hits"]["hits"]

    results = []
    for hit in hits:
        source = hit["_source"]
        original_example = source["original_example"]

        # 检查是否真正匹配
        if original_example in text or text in original_example:
            results.append(
                {
                    "group_id": source["group_id"],
                    "group_name": source["group_name"],
                    "group_type": source["group_type"],
                    "intent_id": source["intent_id"],
                    "intent_name": source["intent_name"],
                    "intent_type": source["intent_type"],
                    "probability": 1.0,  # 完全匹配的分值为1
                    "matched_examples": [original_example],
                    "_score": hit["_score"],
                }
            )

    return results


async def vector_classification_search(
    text: str, text_embedding: list[float], group_id: str, k: int = 100
) -> list[dict[str, Any]]:
    """分类的向量相似度搜索"""
    client = es_client.client

    query = {
        "knn": {
            "field": "embedding",
            "query_vector": text_embedding,
            "k": k,
            "num_candidates": k * 2,  # 通常设置为 k 的 2-10 倍
            "filter": {"term": {"group_id": group_id}},
        }
    }

    response = await client.search(index=CLASSIFICATION_INDEX, body=query)
    hits = response["hits"]["hits"]

    results = []
    for hit in hits:
        source = hit["_source"]
        # KNN 搜索的分值直接来自相似度计算
        similarity_score = hit["_score"]

        results.append(
            {
                "group_id": source["group_id"],
                "group_name": source["group_name"],
                "group_type": source["group_type"],
                "intent_id": source["intent_id"],
                "intent_name": source["intent_name"],
                "intent_type": source["intent_type"],
                "original_example": source["original_example"],
                "similarity_score": similarity_score,
                "_score": hit["_score"],
            }
        )

    return results


class MatchSensitiveWordsResponse(APIResponse):
    class Item(BaseModel):
        group_id: str
        group_name: str
        probability: float = 1.0
        matched_terms: list[str]

    data: list[Item] = Field(default_factory=list)

    model_config: ClassVar = {
        "json_schema_extra": {
            "examples": [
                {
                    "code": 200,
                    "msg": "success",
                    "data": [
                        {
                            "group_id": "1",
                            "group_name": "敏感词",
                            "probability": 1.0,
                            "matched_terms": ["投诉"],
                        }
                    ],
                }
            ]
        }
    }


async def _filter_exact_results_with_rerank(
    text: str, exact_results: list[dict]
) -> list[dict]:
    """对完全匹配结果使用reranker模型过滤"""
    if not exact_results:
        return []

    # 准备重排序的文档列表
    documents = []
    group_names = []
    for result in exact_results:
        # 使用matched_terms中的第一个词作为文档
        if result["matched_terms"]:
            documents.append(result["matched_terms"][0])
            group_names.append(result["group_name"])

    if not documents:
        return []

    tasks = [
        rerank([text, f"{group_name}：{text}"], doc)
        for doc, group_name in zip(documents, group_names, strict=False)
    ]
    # 使用排序模型重新排序
    rerank_results = await asyncio.gather(*tasks)

    # 过滤结果
    filtered_results = []
    group_names = set()
    for index, (rerank_result, rerank_result_with_group_name) in enumerate(
        rerank_results
    ):
        # 找到对应的原始结果
        original_result = exact_results[index]

        # 使用重排序模型的分值作为过滤条件
        min_score = min(
            rerank_result.relevance_score, rerank_result_with_group_name.relevance_score
        )

        max_score = max(
            rerank_result.relevance_score, rerank_result_with_group_name.relevance_score
        )

        # 设置阈值为0.5
        if (
            min_score >= 0.6
            or max_score >= 0.7
            or original_result["group_name"] in group_names
        ):
            # 保持原始结果，但更新概率分数
            filtered_result = original_result.copy()
            # filtered_result["probability"] = min_score
            filtered_results.append(filtered_result)
            group_names.add(filtered_result["group_name"])

    return filtered_results


async def _perform_semantic_search(
    text: str,
    group_type: str,
    semantic_score_threshold: float = 0.99,
    semantic_group_threshold: float = 0.9,
) -> list[dict]:
    """执行语义向量搜索"""
    # 生成查询文本的向量嵌入
    text_embedding = await embedding(text)

    # 向量相似度搜索
    vector_candidates = await vector_search(text, text_embedding, group_type, k=3)

    if not vector_candidates:
        return []

    # 准备重排序的文档列表
    documents = [candidate["original_word"] for candidate in vector_candidates]
    group_names = [candidate["group_name"] for candidate in vector_candidates]

    tasks = [
        rerank([text, f"{group_name}：{text}"], doc)
        for doc, group_name in zip(documents, group_names, strict=False)
    ]
    # 使用排序模型重新排序
    rerank_results = await asyncio.gather(*tasks)

    # 将重排序结果与原始候选结果合并
    semantic_results = []
    for index, (rerank_result, rerank_result_with_group_name) in enumerate(
        rerank_results
    ):
        # 找到对应的原始候选
        original_candidate = vector_candidates[index]

        # 过滤掉分值过低的结果(可以调整阈值)
        print(
            rerank_result.relevance_score, rerank_result_with_group_name.relevance_score
        )
        if (
            rerank_result.relevance_score >= semantic_score_threshold
            and rerank_result_with_group_name.relevance_score > semantic_group_threshold
        ):  # 阈值可以根据实际情况调整
            semantic_results.append(
                {
                    "group_id": original_candidate["group_id"],
                    "group_name": original_candidate["group_name"],
                    "probability": semantic_score_threshold,
                    "matched_terms": [original_candidate["original_word"]],
                    "search_type": "semantic",
                }
            )

    return semantic_results


def _merge_search_results(
    exact_results: list[dict], vector_results: list[dict]
) -> list[dict]:
    """合并完全匹配和语义搜索的结果"""
    # 使用字典来去重, 以group_id为key
    merged_dict = {}

    # 先处理完全匹配结果(优先级更高)
    for result in exact_results:
        group_id = result["group_id"]
        if group_id not in merged_dict:
            merged_dict[group_id] = {
                "group_id": group_id,
                "group_name": result["group_name"],
                "probability": result["probability"],  # 完全匹配分值为1.0
                "matched_terms": result["matched_terms"],
                "search_type": "exact",
            }
        else:
            # 合并匹配的词汇
            merged_dict[group_id]["matched_terms"].extend(result["matched_terms"])
            # 去重
            merged_dict[group_id]["matched_terms"] = list(
                set(merged_dict[group_id]["matched_terms"])
            )

    # 处理语义搜索结果
    for result in vector_results:
        group_id = result["group_id"]
        if group_id not in merged_dict:
            # 如果没有完全匹配, 则使用语义匹配结果
            merged_dict[group_id] = result
        else:
            # 如果已经有完全匹配结果, 保持完全匹配的分值, 但合并匹配词汇
            merged_dict[group_id]["matched_terms"].extend(result["matched_terms"])
            # 去重
            merged_dict[group_id]["matched_terms"] = list(
                set(merged_dict[group_id]["matched_terms"])
            )

    # 按概率降序排序
    results = list(merged_dict.values())
    results.sort(key=lambda x: x["probability"], reverse=True)

    return results


router = APIRouter()


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
                    "delete_old": True,
                }
            ]
        }
    )
    method: str = Field(description="固定为 update")
    group_id: str = Field(description="敏感词组ID")
    group_name: str = Field(description="敏感词组名称")
    group_type: str = Field(description="1市民，2话务员")
    data: list[str] = Field(description="敏感词列表")
    regex: bool = Field(default=False, description="是否正则表达式")
    delete_old: bool = Field(default=True, description="是否删除旧数据")


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


@router.post(
    "/hitachi_elevator/sensitive/train",
    response_model=APIResponse[None],
    summary="敏感词训练",
    tags=["敏感词"],
)
async def train_sensitive_words(req: SensitiveTrainRequest) -> APIResponse:
    """训练敏感词数据"""
    if not req.data:
        return APIResponse(code=400, msg="数据不能为空")

    try:
        # 并行生成所有敏感词的向量嵌入
        embeddings = await asyncio.gather(*(embedding(word) for word in req.data))

        # 索引到Elasticsearch
        await index_sensitive_words(
            group_id=req.group_id,
            group_name=req.group_name,
            group_type=req.group_type,
            words=req.data,
            embeddings=embeddings,
            is_regex=req.regex,
            delete_old=req.delete_old,
        )

        logger.info(f"成功训练敏感词组 {req.group_id}: {len(req.data)} 个词汇")
        return APIResponse(msg="训练成功")

    except Exception as e:
        logger.exception(f"训练敏感词失败: {e!s}")
        return APIResponse(code=500, msg=f"训练失败: {e!s}")


@router.post(
    "/hitachi_elevator/sensitive/delete",
    response_model=APIResponse[None],
    summary="敏感词删除",
    tags=["敏感词"],
)
async def delete_sensitive_words(req: SensitiveDeleteRequest) -> APIResponse:
    """删除敏感词数据"""
    if not req.group_id:
        return APIResponse(code=400, msg="group_id不能为空")

    try:
        await delete_by_group_ids(req.group_id)
        logger.info(f"成功删除敏感词组: {req.group_id}")
        return APIResponse(msg="删除成功")

    except Exception as e:
        logger.exception(f"删除敏感词失败: {e!s}")
        return APIResponse(code=500, msg=f"删除失败: {e!s}")


@router.post(
    "/hitachi_elevator/sensitive/match",
    response_model=APIResponse[list[SensitiveMatchItem]],
    summary="敏感词匹配",
    tags=["敏感词"],
)
async def match_sensitive_words(
    req: SensitiveMatchRequest,
) -> MatchSensitiveWordsResponse:
    """匹配敏感词"""
    if not req.text.strip():
        return MatchSensitiveWordsResponse(code=400, msg="文本不能为空")

    # 同时进行完全匹配和语义向量匹配
    exact_results, vector_results = await asyncio.gather(
        exact_match_search(req.text, req.type),
        _perform_semantic_search(
            req.text,
            req.type,
            req.semantic_score_threshold or 0.99,
            req.semantic_group_threshold or 0.9,
        ),
    )

    # 根据配置决定是否使用reranker过滤完全匹配结果
    if req.use_rerank_filter_full_match:
        filtered_exact_results = await _filter_exact_results_with_rerank(
            req.text, exact_results
        )
    else:
        filtered_exact_results = exact_results

    # 合并和去重结果
    merged_results = _merge_search_results(filtered_exact_results, vector_results)

    # 转换为响应格式
    response_items = []
    for result in merged_results:
        response_items.append(
            MatchSensitiveWordsResponse.Item(
                group_id=result["group_id"],
                group_name=result["group_name"],
                probability=result["probability"],
                matched_terms=result["matched_terms"],
            )
        )

    logger.info(f"敏感词匹配完成, 找到 {len(response_items)} 个匹配结果")
    return MatchSensitiveWordsResponse(data=response_items)


async def run_integration_tests(
    api_url: str | None = None, use_test_client: bool = False, verbose: bool = False
) -> None:
    """Run sensitive integration tests (minimal sanity checks)."""
    import httpx
    import click

    if verbose:
        click.echo(f"\n{'=' * 50}")
        click.echo("Running Sensitive Integration Tests")
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
            "name": "basic match",
            "path": "/hitachi_elevator/sensitive/match",
            "payload": {
                "type": "1",
                "text": "投诉",
                "use_rerank_filter_full_match": False,
            },
        }
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        try:
            if verbose:
                click.echo(f"\nTest {i}: {test_case['name']}")

            response = await client.post(
                test_case["path"], json=test_case["payload"], timeout=60
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
                    snippet = response.text[:200].replace("\n", " ")
                    click.echo(
                        f"  ✗ Failed - Status: {response.status_code}, Body: {snippet}"
                    )
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
