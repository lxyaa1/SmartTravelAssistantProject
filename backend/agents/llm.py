from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from backend.schemas.trip import McpResults, TripPlan, TripRequest, ValidationIssue


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"

T = TypeVar("T", bound=BaseModel)


def should_use_llm(state: dict[str, Any]) -> bool:
    explicit = state.get("use_llm")
    if explicit is not None:
        return bool(explicit)

    env_value = os.getenv("TRAVEL_AGENT_USE_LLM")
    if env_value is not None:
        return env_value.strip().lower() not in {"0", "false", "no", "off"}

    return bool(os.getenv("DASHSCOPE_API_KEY"))


def generate_initial_plan_with_llm(request: TripRequest, mcp_results: McpResults) -> TripPlan:
    prompt = {
        "task": "Generate an initial structured travel itinerary.",
        "rules": [
            "Return only a JSON object that matches TripPlan.",
            "Do not include markdown or explanatory text.",
            "Every day must contain visits ordered by sequence.",
            "Use candidate attractions, weather, and accommodation areas from MCP results.",
            "Must-visit places must be included unless impossible.",
            "If a day has heavy rain, prefer indoor attractions for that date.",
            "If children or infants are present, keep the schedule relaxed and avoid late activities.",
            "Use accommodation_area from MCP accommodation areas when available.",
            "Set transport_to_next to null for the last visit of each day.",
        ],
        "allowed_values": _allowed_values(),
        "request": request.model_dump(mode="json"),
        "mcp_results": mcp_results.model_dump(mode="json"),
        "trip_plan_schema": TripPlan.model_json_schema(),
    }
    return _chat_structured(
        system_prompt="You are a precise travel-planning engine. You output valid JSON only.",
        user_payload=prompt,
        output_model=TripPlan,
        temperature=0.2,
    )


def replan_with_llm(
    request: TripRequest,
    current_plan: TripPlan,
    issues: list[ValidationIssue],
    mcp_results: McpResults,
) -> TripPlan:
    prompt = {
        "task": "Revise the structured travel itinerary to address validation issues.",
        "rules": [
            "Return only a JSON object that matches TripPlan.",
            "Do not include markdown or explanatory text.",
            "Prefer local edits over rewriting the whole itinerary.",
            "Preserve must-visit places unless an issue makes them impossible.",
            "Prefer swapping dates, reordering nearby places, or replacing non-must-visit places.",
            "Use MCP results and validation issues as the source of truth.",
            "Set transport_to_next to null for the last visit of each day.",
        ],
        "allowed_values": _allowed_values(),
        "request": request.model_dump(mode="json"),
        "current_plan": current_plan.model_dump(mode="json"),
        "issues": [issue.model_dump(mode="json") for issue in issues],
        "mcp_results": mcp_results.model_dump(mode="json"),
        "trip_plan_schema": TripPlan.model_json_schema(),
    }
    return _chat_structured(
        system_prompt="You are a precise travel replanning engine. You output valid JSON only.",
        user_payload=prompt,
        output_model=TripPlan,
        temperature=0.1,
    )


def _chat_structured(
    system_prompt: str,
    user_payload: dict[str, Any],
    output_model: type[T],
    temperature: float,
) -> T:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is required when LLM mode is enabled")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL),
    )
    completion = client.chat.completions.create(
        model=os.getenv("DASHSCOPE_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Please output a standard JSON object. "
                    "The JSON must satisfy the provided schema.\n\n"
                    f"{json.dumps(user_payload, ensure_ascii=False)}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("DashScope returned an empty response")

    try:
        return output_model.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(f"DashScope response did not match {output_model.__name__}: {exc}") from exc


def _allowed_values() -> dict[str, list[str]]:
    return {
        "place_category": ["outdoor", "indoor", "food", "culture", "shopping", "hotel_area"],
        "transport_mode": ["walk", "taxi", "transit", "train", "flight"],
    }
