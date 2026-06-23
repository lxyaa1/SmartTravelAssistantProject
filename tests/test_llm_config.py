from __future__ import annotations

from backend.agents.llm import _repair_payload_for_schema, should_use_llm
from backend.schemas.trip import CityRoutePlan, TripPlan


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


def test_repair_city_route_plan_payload_normalizes_combined_transport_mode() -> None:
    raw = """
    {
      "origin": "Beijing",
      "destination": "Shanxi",
      "stays": [
        {
          "sequence": 1,
          "city": "Xinzhou",
          "start_date": "2026-07-01",
          "end_date": "2026-07-02",
          "anchor_places": ["Wutai Mountain"],
          "lodging_anchor": "Wutai Mountain"
        }
      ],
      "segments": [
        {
          "sequence": 1,
          "segment_type": "outbound",
          "origin": "Beijing",
          "destination": "Xinzhou",
          "origin_city": "Beijing",
          "destination_city": "Xinzhou",
          "mode": "taxi + train"
        }
      ]
    }
    """

    repaired = _repair_payload_for_schema(raw, CityRoutePlan)
    plan = CityRoutePlan.model_validate(repaired)

    assert plan.segments[0].mode.value == "train"


def test_repair_trip_plan_payload_splits_cross_midnight_sleep_timeline_items() -> None:
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
          "timeline": [
            {
              "sequence": 1,
              "item_type": "stay",
              "start_time": "22:30",
              "end_time": "07:30",
              "city": "Xinzhou",
              "stay": {
                "place_name": "Xinzhou Hotel",
                "city": "Xinzhou",
                "purpose": "sleep"
              }
            },
            {
              "sequence": 2,
              "item_type": "stay",
              "start_time": "12:00",
              "end_time": "12:00",
              "city": "Xinzhou",
              "stay": {
                "place_name": "Buffer",
                "city": "Xinzhou",
                "purpose": "rest"
              }
            }
          ]
        }
      ]
    }
    """

    repaired = _repair_payload_for_schema(raw, TripPlan)
    plan = TripPlan.model_validate(repaired)

    timeline = plan.days[0].timeline
    assert [(item.start_time.hour, item.end_time.hour) for item in timeline[:2]] == [(0, 7), (12, 12)]
    assert all(item.end_time > item.start_time for item in timeline)


def test_repair_trip_plan_payload_removes_timeline_overlaps() -> None:
    raw = """
    {
      "title": "test",
      "origin": "Beijing",
      "destination": "Shanxi",
      "days": [
        {
          "day": 1,
          "date": "2026-07-01",
          "city": "Taiyuan",
          "timeline": [
            {
              "sequence": 1,
              "item_type": "stay",
              "start_time": "09:00",
              "end_time": "11:00",
              "city": "Taiyuan",
              "stay": {
                "place_name": "Jinci Temple",
                "city": "Taiyuan",
                "purpose": "visit"
              }
            },
            {
              "sequence": 2,
              "item_type": "move",
              "start_time": "10:30",
              "end_time": "11:00",
              "city": "Taiyuan",
              "move": {
                "origin": "Jinci Temple",
                "destination": "Hotel",
                "mode": "taxi",
                "purpose": "transfer"
              }
            }
          ]
        }
      ]
    }
    """

    repaired = _repair_payload_for_schema(raw, TripPlan)
    plan = TripPlan.model_validate(repaired)

    timeline = plan.days[0].timeline
    assert timeline[1].start_time >= timeline[0].end_time
    assert timeline[1].move is not None
    assert timeline[1].move.purpose.value == "local"


def test_repair_trip_plan_payload_maps_shopping_purpose_to_visit_category() -> None:
    raw = """
    {
      "title": "test",
      "origin": "Beijing",
      "destination": "Shanxi",
      "days": [
        {
          "day": 1,
          "date": "2026-07-01",
          "city": "Taiyuan",
          "timeline": [
            {
              "sequence": 1,
              "item_type": "stay",
              "start_time": "15:00",
              "end_time": "16:00",
              "city": "Taiyuan",
              "stay": {
                "place_name": "Liuxiang Shopping Street",
                "city": "Taiyuan",
                "purpose": "shopping"
              }
            }
          ]
        }
      ]
    }
    """

    repaired = _repair_payload_for_schema(raw, TripPlan)
    plan = TripPlan.model_validate(repaired)

    stay = plan.days[0].timeline[0].stay
    assert stay is not None
    assert stay.purpose.value == "visit"
    assert stay.category is not None
    assert stay.category.value == "shopping"


def test_repair_trip_plan_payload_clamps_overflow_times_and_repairs_enums() -> None:
    raw = """
    {
      "title": "test",
      "origin": "Beijing",
      "destination": "Shanxi",
      "days": [
        {
          "day": 1,
          "date": "2026-07-01",
          "city": "Taiyuan",
          "timeline": [
            {
              "sequence": 1,
              "item_type": "stay",
              "start_time": "09:00",
              "end_time": "10:00",
              "city": "Taiyuan",
              "stay": {
                "place_name": "Liuxiang Shopping Street",
                "city": "Taiyuan",
                "purpose": "shopping"
              }
            },
            {
              "sequence": 2,
              "item_type": "move",
              "start_time": "25:52",
              "end_time": "26:52",
              "city": "Taiyuan",
              "move": {
                "origin": "Liuxiang Shopping Street",
                "destination": "Hotel",
                "mode": "taxi",
                "purpose": "transfer"
              }
            }
          ]
        },
        {
          "day": 2,
          "date": "2026-07-02",
          "city": "Taiyuan",
          "timeline": [
            {
              "sequence": 1,
              "item_type": "stay",
              "start_time": "09:00",
              "end_time": "11:00",
              "city": "Taiyuan",
              "stay": {
                "place_name": "Jinci Temple",
                "city": "Taiyuan",
                "purpose": "visit"
              }
            },
            {
              "sequence": 2,
              "item_type": "stay",
              "start_time": "10:30",
              "end_time": "12:00",
              "city": "Taiyuan",
              "stay": {
                "place_name": "Lunch",
                "city": "Taiyuan",
                "purpose": "food"
              }
            }
          ]
        }
      ]
    }
    """

    repaired = _repair_payload_for_schema(raw, TripPlan)
    plan = TripPlan.model_validate(repaired)

    day1 = plan.days[0]
    assert day1.timeline[0].stay is not None
    assert day1.timeline[0].stay.purpose.value == "visit"
    assert day1.timeline[0].stay.category is not None
    assert day1.timeline[0].stay.category.value == "shopping"
    assert day1.timeline[1].end_time.hour <= 23
    assert day1.timeline[1].move is not None
    assert day1.timeline[1].move.purpose.value == "local"

    day2 = plan.days[1]
    assert day2.timeline[1].start_time >= day2.timeline[0].end_time
    assert day2.timeline[1].stay is not None
    assert day2.timeline[1].stay.purpose.value == "meal"


def test_repair_trip_plan_payload_normalizes_combined_transport_modes() -> None:
    raw = """
    {
      "title": "test",
      "origin": "Beijing",
      "destination": "Shanxi",
      "route_segments": [
        {
          "sequence": 1,
          "segment_type": "outbound",
          "origin": "Beijing",
          "destination": "Xinzhou",
          "origin_city": "Beijing",
          "destination_city": "Xinzhou",
          "mode": "taxi + train"
        }
      ],
      "days": [
        {
          "day": 1,
          "date": "2026-07-01",
          "city": "Xinzhou",
          "timeline": [
            {
              "sequence": 1,
              "item_type": "move",
              "start_time": "08:00",
              "end_time": "12:00",
              "city": "Xinzhou",
              "move": {
                "origin": "Beijing",
                "destination": "Xinzhou",
                "origin_city": "Beijing",
                "destination_city": "Xinzhou",
                "mode": "taxi + train",
                "purpose": "outbound"
              }
            }
          ]
        }
      ]
    }
    """

    repaired = _repair_payload_for_schema(raw, TripPlan)
    plan = TripPlan.model_validate(repaired)

    assert plan.route_segments[0].mode.value == "train"
    assert plan.days[0].timeline[0].move is not None
    assert plan.days[0].timeline[0].move.mode.value == "train"
