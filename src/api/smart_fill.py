import asyncio
import io
import json
import os
from typing import Literal
import traceback

import click
import httpx
import openai
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from ..models import APIResponse

router = APIRouter()

client = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


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


def get_default_fill_data() -> SmartFillData:
    return SmartFillData()


async def extract_fields_from_conversation(
    conversation: str, call_type: str
) -> SmartFillData:
    instruction = (
        "这是一个电梯故障维修的客服对话。请从对话中提取以下信息，返回JSON格式。\n"
    )
    if call_type == "mid":
        instruction += "注意：这是中间过程调用，如果某个字段置信度低于70%或无法确定，请留空不要猜测。\n"
    else:
        instruction += (
            "注意：这是结束通话调用，请尽量为每个字段提供结果，即使置信度较低。\n"
        )

    instruction += """
需要提取的字段：
1. complaint_content: 咨询/投诉内容 - 总结用户反映的主要问题
2. address: 所在地区地址 - 提取完整的省市区街道或小区地址
3. org_name: 使用单位名称 - 电梯使用单位或物业公司名称
4. contact_name: 联系人姓名 - 联系人的姓名
5. contact_phone: 联系人电话 - 联系人的电话号码

对于每个字段，需要返回：
- value: 提取的结果值
- evidence: 原文中支持该提取结果的关键片段

请严格按照以下JSON格式返回，不要包含任何额外说明：
{
  "complaint_content": {"value": "", "evidence": ""},
  "address": {"value": "", "evidence": ""},
  "org_name": {"value": "", "evidence": ""},
  "contact_name": {"value": "", "evidence": ""},
  "contact_phone": {"value": "", "evidence": ""}
}
仅返回上述JSON，不要包含其它文字或前后缀。
"""

    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {
                "role": "system",
                "content": "你是一个专业的信息提取助手，擅长从客服对话中提取关键信息。",
            },
            {
                "role": "user",
                "content": f"{instruction}\n\n对话内容：\n{conversation}",
            },
        ],
        temperature=0.0,
        top_p=0.0,
        response_format={"type": "json_object"},
        extra_body={
            "thinking": {"type": "disabled"},
        },
    )

    result_text = response.choices[0].message.content
    if not result_text:
        return get_default_fill_data()

    result_json = json.loads(result_text)
    return SmartFillData(**result_json)


@router.post(
    "/hitachi_elevator/smart_fill",
    response_model=APIResponse[SmartFillData],
    summary="智能填单",
    tags=["智能填单"],
)
async def smart_fill(request: SmartFillRequest) -> APIResponse[SmartFillData]:
    try:
        conversation = "\n".join(
            f"{turn.role}: {turn.text}" for turn in request.history
        )
        data = await extract_fields_from_conversation(conversation, request.call_type)
        return APIResponse(data=data)
    except Exception as e:
        stack = traceback.format_exc()
        return APIResponse(
            code=500,
            msg=f"{e}\n{stack}",
            data=get_default_fill_data(),
        )


async def run_integration_tests(
    api_url: str | None = None,
    use_test_client: bool = False,
    verbose: bool = False,
) -> None:
    """Run smart_fill integration tests."""
    import io

    if verbose:
        click.echo(f"\n{'=' * 50}")
        click.echo("Running Smart Fill Integration Tests")
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
            "name": "mid call extraction",
            "request": {
                "session_id": "test_session_001",
                "call_type": "mid",
                "turn_id": 5,
                "history": [
                    {"role": "user", "text": "电梯停在12楼了"},
                    {"role": "agent", "text": "请问具体地址？"},
                ],
            },
        },
        {
            "name": "final call extraction",
            "request": {
                "session_id": "test_session_002",
                "call_type": "final",
                "turn_id": 10,
                "history": [
                    {"role": "user", "text": "电梯停在12楼了"},
                    {"role": "agent", "text": "请问具体地址？"},
                    {"role": "user", "text": "天河区体育西路123号"},
                    {"role": "agent", "text": "联系电话？"},
                    {"role": "user", "text": "13800138000"},
                ],
            },
        },
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        try:
            if verbose:
                click.echo(f"\nTest {i}: {test_case['name']}")

            response = await client.post(
                "/hitachi_elevator/smart_fill",
                json=test_case["request"],
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    passed += 1
                    data = result.get("data", {})
                    if verbose:
                        complaint = data.get("complaint_content", {}).get("value", "")
                        address = data.get("address", {}).get("value", "")
                        click.echo(
                            f"  ✓ Passed - complaint_content: {complaint[:30]}, address: {address[:30]}"
                        )
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

    exit_code = 0 if failed == 0 else 1
    if exit_code != 0:
        raise click.ClickException("")
