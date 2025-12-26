import os
import re
from datetime import datetime
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from loguru import logger

# Elasticsearch配置
ES_HOST = os.getenv("ES_HOST", "localhost:9200")
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
