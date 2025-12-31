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


def get_default_trapped_data() -> TrappedDetectData:
    return TrappedDetectData(
        is_trapped=False, event_type="不确定", probability=0.0, evidence=""
    )


async def detect_trapped_event(role: str, text: str) -> TrappedDetectData:
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

要求：probability为0-1之间的浮点数，表示判断的置信度。仅返回上述JSON，不要包含其他文本。"""

    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {
                "role": "system",
                "content": "你是一个专业的电梯困人事件检测助手，需要快速准确判断对话内容是否涉及困人/困梯事件。",
            },
            {
                "role": "user",
                "content": f"{instruction}\n\n对话内容（{role}）：{text}",
            },
        ],
        temperature=0.0,
        top_p=0.0,
        response_format={"type": "json_object"},
    )

    result_text = response.choices[0].message.content
    if not result_text:
        return get_default_trapped_data()

    result_json = json.loads(result_text)
    return TrappedDetectData(**result_json)


@router.post(
    "/hitachi_elevator/trapped_detect",
    response_model=APIResponse[TrappedDetectData],
    summary="困人困梯检测",
    tags=["困人困梯检测"],
)
async def trapped_detect(
    request: TrappedDetectRequest,
) -> APIResponse[TrappedDetectData]:
    try:
        data = await detect_trapped_event(request.role, request.text)
        return APIResponse(data=data)
    except Exception as e:
        stack = traceback.format_exc()
        return APIResponse(
            code=500,
            msg=f"{e}\n{stack}",
            data=get_default_trapped_data(),
        )


async def run_integration_tests(
    api_url: str | None = None,
    use_test_client: bool = False,
    verbose: bool = False,
) -> None:
    """Run trapped_detect integration tests."""
    import click

    if verbose:
        click.echo(f"\n{'=' * 50}")
        click.echo("Running Trapped Detect Integration Tests")
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
            "name": "person trapped detection",
            "request": {
                "session_id": "test_session_001",
                "turn_id": 1,
                "role": "user",
                "text": "电梯突然停了，我被困在里面了，帮忙联系维修！",
            },
        },
        {
            "name": "elevator malfunction",
            "request": {
                "session_id": "test_session_002",
                "turn_id": 2,
                "role": "user",
                "text": "电梯停在12楼了，门打不开。",
            },
        },
        {
            "name": "uncertain scenario",
            "request": {
                "session_id": "test_session_003",
                "turn_id": 3,
                "role": "user",
                "text": "好像有点问题",
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
                "/hitachi_elevator/trapped_detect",
                json=test_case["request"],
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    passed += 1
                    data = result.get("data", {})
                    if verbose:
                        click.echo(
                            f"  ✓ Passed - is_trapped: {data['is_trapped']}, "
                            f"event_type: {data['event_type']}, "
                            f"probability: {data['probability']}"
                        )
                else:
                    failed += 1
                    if verbose:
                        click.echo(
                            f"  ✗ Failed - Code: {result['code']}, Msg: {result['msg']}"
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
