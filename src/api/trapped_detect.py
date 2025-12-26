import json
import os
from typing import Literal

import openai
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from ..models import APIResponse

router = APIRouter()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
def trapped_detect(request: TrappedDetectRequest) -> APIResponse[TrappedDetectData]:
    instruction = """这是一个电梯客服对话，请判断当前文本是否包含"电梯困人事件"。

判断标准：
- 困人：有人被困在电梯内，无法出来
- 困梯：电梯发生故障停运，但无人被困
- 不确定：无法从文本中判断

请返回JSON格式：
{
  "is_trapped": true/false,
  "event_type": "困人"/"困梯"/"不确定",
  "probability": 0.0-1.0,
  "evidence": "触发判断的关键短语"
}

要求：probability为0-1之间的浮点数，表示判断的置信度。"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的电梯困人事件检测助手，需要快速准确判断对话内容是否涉及困人/困梯事件。",
                },
                {
                    "role": "user",
                    "content": f"{instruction}\n\n对话内容（{request.role}）：{request.text}",
                },
            ],
            temperature=0.1,
            max_tokens=200,
        )

        result_text = response.choices[0].message.content
        if not result_text:
            return APIResponse(
                data=TrappedDetectData(
                    is_trapped=False, event_type="不确定", probability=0.0, evidence=""
                )
            )
        result_json = json.loads(result_text)

        return APIResponse(data=TrappedDetectData(**result_json))
    except Exception as e:
        return APIResponse(
            data=TrappedDetectData(
                is_trapped=False, event_type="不确定", probability=0.0, evidence=""
            )
        )
