from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    code: int = Field(default=200, description="业务状态码")
    msg: str = Field(default="success", description="状态说明")
    data: Optional[T] = Field(default=None, description="业务数据")
