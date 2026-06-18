from __future__ import annotations

from datetime import date

from backend.mcp.amap import _extract_pois, _parse_route_metrics, _parse_weather_result


def test_parse_official_amap_weather_content_block() -> None:
    raw = [
        {
            "type": "text",
            "text": (
                '{"city":"杭州市","forecasts":[{"date":"2026-06-18",'
                '"dayweather":"中雨","nightweather":"中雨","daytemp":"32","nighttemp":"24"}]}'
            ),
        }
    ]

    result = _parse_weather_result(raw=raw, city="杭州", query_date=date(2026, 6, 18))

    assert result.city == "杭州"
    assert result.date == date(2026, 6, 18)
    assert result.condition == "heavy rain"
    assert "中雨" in (result.warning or "")


def test_extract_official_amap_pois_from_content_block() -> None:
    raw = [
        {
            "type": "text",
            "text": (
                '{"suggestion":{"keywords":"","ciytes":{"suggestion":[]}},'
                '"pois":[{"id":"B023B13L9M","name":"杭州西湖风景名胜区","typecode":"110202"}]}'
            ),
        }
    ]

    pois = _extract_pois(raw)

    assert pois == [{"id": "B023B13L9M", "name": "杭州西湖风景名胜区", "typecode": "110202"}]


def test_parse_official_amap_route_metrics_from_paths() -> None:
    raw = [
        {
            "type": "text",
            "text": '{"paths":[{"distance":"2340","duration":"768"}]}',
        }
    ]

    duration_minutes, distance_km = _parse_route_metrics(raw)

    assert duration_minutes == 13
    assert distance_km == 2.34
