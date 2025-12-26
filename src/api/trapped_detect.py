from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from ..models import APIResponse

router = APIRouter()


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


@router.post(
    "/hitachi_elevator/trapped_detect",
    response_model=APIResponse[TrappedDetectData],
    summary="困人困梯检测",
    tags=["困人困梯检测"],
)
def trapped_detect(_: TrappedDetectRequest) -> APIResponse[TrappedDetectData]:
    data = TrappedDetectData(
        is_trapped=False,
        event_type="不确定",
        probability=0.0,
        evidence="",
    )
    return APIResponse(data=data)
