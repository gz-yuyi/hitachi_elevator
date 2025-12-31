import os

from elasticsearch import AsyncElasticsearch


def normalize_host(raw_host: str) -> str:
    if raw_host.startswith("http://") or raw_host.startswith("https://"):
        return raw_host
    return f"http://{raw_host}"


# Elasticsearch配置
ES_HOST = normalize_host(os.getenv("ES_HOST", "localhost:9200"))
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "changeme")


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
