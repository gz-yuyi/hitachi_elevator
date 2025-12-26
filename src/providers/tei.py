import os

import aiohttp

from . import RerankResult

__all__ = ["embedding", "rerank"]


# Separate URLs for embedding and rerank
EMBEDDING_URL = os.getenv("TIE_EMBEDDING_URL", "http://tie-embedding:80")
RERANK_URL = os.getenv("TIE_RERANK_URL", "http://tie-rerank:80")

# TEI 通常不需要认证，但如果需要可以设置 API_KEY
API_KEY = os.getenv("TEI_API_KEY")
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"


async def embedding(input_text: str) -> list[float]:
    """
    调用 TEI 的 embedding 接口获取文本向量

    Args:
        input_text: 输入文本

    Returns:
        向量列表
    """
    url = f"{EMBEDDING_URL}/embed"
    payload = {"inputs": input_text}

    async with (
        aiohttp.ClientSession() as session,
        session.post(url, json=payload, headers=HEADERS) as response,
    ):
        response_data = await response.json()

        # TEI 返回的格式可能是直接的向量数组，或者包装在某个字段中
        # 根据文档，应该直接返回向量数组
        if isinstance(response_data, list) and len(response_data) > 0:
            # 如果是列表，取第一个元素（单个文本的情况）
            return (
                response_data[0]
                if isinstance(response_data[0], list)
                else response_data
            )

        # 兼容性处理：如果返回格式不同，尝试其他可能的字段
        if isinstance(response_data, dict):
            if "embeddings" in response_data:
                return response_data["embeddings"][0]
            elif "data" in response_data:
                return response_data["data"][0]

        return response_data


async def rerank(documents: list[str], query: str) -> list[RerankResult]:
    """
    调用 TEI 的 rerank 接口对文档进行重新排序

    Args:
        documents: 文档列表
        query: 查询文本

    Returns:
        重新排序的结果列表
    """
    url = f"{RERANK_URL}/rerank"
    payload = {
        "query": query,
        "texts": documents,
        "raw_scores": False,  # 使用标准化分数
    }

    async with (
        aiohttp.ClientSession() as session,
        session.post(url, json=payload, headers=HEADERS) as response,
    ):
        response_data = await response.json()

        # TEI rerank 返回格式应该是包含 results 字段的字典
        if isinstance(response_data, dict) and "results" in response_data:
            results = response_data["results"]
        elif isinstance(response_data, list):
            results = response_data
        else:
            results = []

        # 转换为标准的 RerankResult 格式
        rerank_results = []
        for result in results:
            rerank_result = RerankResult(
                document=RerankResult.Document(
                    text=result.get("text", documents[result.get("index", 0)])
                ),
                index=result.get("index", 0),
                relevance_score=result.get("relevance_score", result.get("score", 0.0)),
            )
            rerank_results.append(rerank_result)

        return rerank_results
