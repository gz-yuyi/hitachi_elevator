from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from ..models import APIResponse

router = APIRouter()


class ChatTurn(BaseModel):
    role: Literal["user", "agent"] = Field(description="说话角色：user/agent")
    text: str = Field(description="ASR转写文本")


class FillField(BaseModel):
    value: str = Field(default="", description="抽取结果")
    evidence: str = Field(default="", description="命中原文片段")


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


class SmartFillData(BaseModel):
    complaint_content: FillField = Field(
        default_factory=FillField, description="咨询/投诉内容"
    )
    address: FillField = Field(default_factory=FillField, description="所在地区地址")
    org_name: FillField = Field(default_factory=FillField, description="使用单位名称")
    contact_name: FillField = Field(default_factory=FillField, description="联系人姓名")
    contact_phone: FillField = Field(
        default_factory=FillField, description="联系人电话"
    )


@router.post(
    "/hitachi_elevator/smart_fill",
    response_model=APIResponse[SmartFillData],
    summary="智能填单",
    tags=["智能填单"],
)
def smart_fill(_: SmartFillRequest) -> APIResponse[SmartFillData]:
    return APIResponse(data=SmartFillData())
