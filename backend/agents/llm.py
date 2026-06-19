from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from backend.schemas.trip import CityRoutePlan, McpResults, TripPlan, TripRequest, ValidationIssue


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


def generate_city_route_plan_with_llm(request: TripRequest) -> CityRoutePlan:
    prompt = {
        "task": "Create a structured city and region route skeleton before detailed day planning.",
        "rules": [
            "Return only a JSON object that matches CityRoutePlan.",
            "Do not include markdown or explanatory text.",
            "Always include an outbound segment from request.origin to the first stay city when they differ.",
            "Always include a return segment from the last stay city back to request.origin when they differ.",
            "If request.destination is a broad region or province, infer concrete base cities from must-visit places.",
            "For every stay, set lodging_anchor to the primary attraction or area that accommodation should be near.",
            "Do not invent train or flight numbers; leave train_or_flight_number empty unless present in input.",
        ],
        "allowed_values": _allowed_values(),
        "request": request.model_dump(mode="json"),
        "city_route_plan_schema": CityRoutePlan.model_json_schema(),
    }
    return _chat_structured(
        system_prompt="You are a precise route-skeleton planner. You output valid JSON only.",
        user_payload=prompt,
        output_model=CityRoutePlan,
        temperature=0.1,
    )


def generate_initial_plan_with_llm(
    request: TripRequest,
    mcp_results: McpResults,
    city_route_plan: CityRoutePlan | None = None,
) -> TripPlan:
    prompt = {
        "task": "Generate a detailed structured travel itinerary from the city route skeleton and MCP data.",
        "rules": [
            "Return only a JSON object that matches TripPlan.",
            "Do not include markdown or explanatory text.",
            "Populate route_segments from the city_route_plan and include an explicit return segment.",
            "Populate accommodations with concrete hotel or lodging candidates near each stay's lodging_anchor.",
            "Each day must include schedule_blocks that cover the main 24-hour timeline: sleep, meals, transfers, visits, rest, and check-in/check-out where relevant.",
            "A schedule block must never cross midnight because the schema only has time fields. Split sleep into 00:00-07:30 and 22:30-23:59 instead of writing 22:30-07:30.",
            "Every schedule block must have end_time later than start_time on the same day.",
            "Every day must contain visits ordered by sequence.",
            "Use candidate attractions, weather, and accommodation areas from MCP results.",
            "Must-visit places must be included unless impossible.",
            "If a day has heavy rain, prefer indoor attractions for that date.",
            "If children or infants are present, keep the schedule relaxed and avoid late activities.",
            "Use MCP lodging results near anchor places before generic accommodation areas.",
            "For day 1, include arrival_transfer from request.origin to request.destination when they differ.",
            "For the last day, include departure_transfer back to request.origin when origin and destination differ.",
            "For every day with accommodation_area and visits, include start_transfer_to_first and return_transfer_to_accommodation.",
            "Do not invent optimistic route durations; use MCP route results when available and otherwise choose conservative estimates.",
            "Do not use province-level endpoints when a concrete city or attraction anchor is available.",
            "Set transport_to_next to null for the last visit of each day.",
        ],
        "allowed_values": _allowed_values(),
        "request": request.model_dump(mode="json"),
        "city_route_plan": city_route_plan.model_dump(mode="json") if city_route_plan else None,
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
            "Preserve and correct route_segments, accommodations, schedule_blocks, arrival_transfer, departure_transfer, start_transfer_to_first, return_transfer_to_accommodation, and transport_to_next.",
            "A schedule block must never cross midnight because the schema only has time fields. Split sleep into 00:00-07:30 and 22:30-23:59 instead of writing 22:30-07:30.",
            "Every schedule block must have end_time later than start_time on the same day.",
            "Do not leave impossible 30-40 minute transfers when validation reports a much longer MCP route.",
            "If lodging is too far from attractions, choose a lodging result nearer to the affected anchor place.",
            "If a return transfer is missing, add it explicitly.",
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
        repaired_payload = _repair_payload_for_schema(content=content, output_model=output_model)
        if repaired_payload is not None:
            try:
                return output_model.model_validate(repaired_payload)
            except ValidationError:
                pass
        raise ValueError(f"DashScope response did not match {output_model.__name__}: {exc}") from exc


def _repair_payload_for_schema(content: str, output_model: type[T]) -> dict[str, Any] | None:
    if output_model is not TripPlan:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _repair_trip_plan_payload(payload)


def _repair_trip_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    repaired = deepcopy(payload)
    days = repaired.get("days")
    if not isinstance(days, list):
        return repaired
    for day in days:
        if not isinstance(day, dict):
            continue
        blocks = day.get("schedule_blocks")
        if not isinstance(blocks, list):
            continue
        day["schedule_blocks"] = _repair_schedule_blocks(blocks)
    return repaired


def _repair_schedule_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    repaired_blocks: list[dict[str, Any]] = []
    for raw_block in blocks:
        if not isinstance(raw_block, dict):
            continue
        block = deepcopy(raw_block)
        start_minutes = _time_to_minutes(block.get("start_time"))
        end_minutes = _time_to_minutes(block.get("end_time"))
        if start_minutes is None or end_minutes is None:
            repaired_blocks.append(block)
            continue
        if end_minutes > start_minutes:
            repaired_blocks.append(block)
            continue

        if str(block.get("block_type", "")).lower() == "sleep" and end_minutes > 0:
            morning = deepcopy(block)
            morning["start_time"] = "00:00"
            morning["end_time"] = _minutes_to_time(end_minutes)
            repaired_blocks.append(morning)

            night = deepcopy(block)
            night["start_time"] = _minutes_to_time(start_minutes)
            night["end_time"] = "23:59"
            repaired_blocks.append(night)
            continue

        block["end_time"] = _minutes_to_time(min(23 * 60 + 59, start_minutes + 15))
        if _time_to_minutes(block["end_time"]) <= start_minutes:
            block["start_time"] = _minutes_to_time(max(0, start_minutes - 15))
            block["end_time"] = _minutes_to_time(start_minutes)
        repaired_blocks.append(block)

    repaired_blocks.sort(key=lambda item: (_time_to_minutes(item.get("start_time")) or 0, int(item.get("sequence", 0) or 0)))
    for index, block in enumerate(repaired_blocks, start=1):
        block["sequence"] = index
    return repaired_blocks


def _time_to_minutes(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return hour * 60 + minute


def _minutes_to_time(minutes: int) -> str:
    minutes = max(0, min(23 * 60 + 59, minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _allowed_values() -> dict[str, list[str]]:
    return {
        "place_category": ["outdoor", "indoor", "food", "culture", "shopping", "hotel_area"],
        "transport_mode": ["walk", "taxi", "transit", "train", "flight"],
        "segment_type": ["outbound", "intercity", "return", "local"],
        "schedule_block_type": [
            "sleep",
            "breakfast",
            "lunch",
            "dinner",
            "rest",
            "visit",
            "local_transfer",
            "intercity_transfer",
            "hotel_checkin",
            "hotel_checkout",
            "free_time",
        ],
    }
