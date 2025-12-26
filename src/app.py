from fastapi import FastAPI

from .api import doc_parse, knowledge, sensitive, smart_fill, trapped_detect

app = FastAPI(title="日立电梯投标Demo算法需求", version="0.1.0")

app.include_router(doc_parse.router)
app.include_router(knowledge.router)
app.include_router(sensitive.router)
app.include_router(smart_fill.router)
app.include_router(trapped_detect.router)
