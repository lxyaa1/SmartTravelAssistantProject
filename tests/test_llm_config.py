from __future__ import annotations

from backend.agents.llm import _repair_payload_for_schema, should_use_llm
from backend.schemas.trip import TripPlan


def test_should_use_llm_can_be_disabled_by_state_even_when_key_exists(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    assert should_use_llm({"use_llm": False}) is False


def test_should_use_llm_uses_dashscope_key_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TRAVEL_AGENT_USE_LLM", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    assert should_use_llm({}) is True


def test_should_use_llm_can_be_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("TRAVEL_AGENT_USE_LLM", "false")

    assert should_use_llm({}) is False


def test_repair_trip_plan_payload_splits_cross_midnight_sleep_blocks() -> None:
    raw = """
    {
      "title": "test",
      "origin": "Beijing",
      "destination": "Shanxi",
      "days": [
        {
          "day": 1,
          "date": "2026-07-01",
          "city": "Xinzhou",
          "schedule_blocks": [
            {
              "sequence": 1,
              "block_type": "sleep",
              "start_time": "22:30",
              "end_time": "07:30",
              "title": "Sleep",
              "city": "Xinzhou"
            },
            {
              "sequence": 2,
              "block_type": "free_time",
              "start_time": "12:00",
              "end_time": "12:00",
              "title": "Buffer",
              "city": "Xinzhou"
            }
          ]
        }
      ]
    }
    """

    repaired = _repair_payload_for_schema(raw, TripPlan)
    plan = TripPlan.model_validate(repaired)

    blocks = plan.days[0].schedule_blocks
    assert [(block.start_time.hour, block.end_time.hour) for block in blocks[:2]] == [(0, 7), (12, 12)]
    assert all(block.end_time > block.start_time for block in blocks)
