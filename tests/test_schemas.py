from __future__ import annotations

from datetime import date, time

import pytest
from pydantic import ValidationError

from backend.schemas.trip import (
    AccommodationRequirement,
    MoveDetail,
    MovePurpose,
    PlaceCategory,
    PlanDay,
    StayDetail,
    StayPurpose,
    TimelineItem,
    TimelineItemType,
    TransportMode,
    TripRequest,
    TravelerGroup,
)


def test_trip_request_rejects_invalid_date_range() -> None:
    with pytest.raises(ValidationError):
        TripRequest(
            origin="Shanghai",
            destination="Hangzhou",
            start_date=date(2026, 7, 3),
            end_date=date(2026, 7, 1),
        )


def test_trip_request_converts_legacy_traveler_count_to_adults() -> None:
    request = TripRequest(
        origin="Shanghai",
        destination="Hangzhou",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 3),
        travelers=2,
    )

    assert request.travelers.adults == 2
    assert request.travelers.total_people == 2
    assert request.travelers.bed_count == 2
    assert request.accommodation is not None
    assert request.accommodation.bed_count == 2


def test_children_count_as_people_without_requiring_extra_beds() -> None:
    request = TripRequest(
        origin="Shanghai",
        destination="Hangzhou",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 3),
        travelers=TravelerGroup(
            adults=2,
            children=1,
            children_need_bed=0,
            children_ages=[6],
        ),
    )

    assert request.travelers.total_people == 3
    assert request.travelers.bed_count == 2
    assert request.accommodation is not None
    assert request.accommodation.bed_count == 2
    assert request.accommodation.allow_children_share_bed is True
    assert request.accommodation.prefer_family_room is True


def test_trip_request_rejects_accommodation_with_too_few_beds() -> None:
    with pytest.raises(ValidationError):
        TripRequest(
            origin="Shanghai",
            destination="Hangzhou",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 3),
            travelers=TravelerGroup(adults=2, children=1, children_need_bed=1),
            accommodation=AccommodationRequirement(bed_count=2),
        )


def test_traveler_group_rejects_more_child_beds_than_children() -> None:
    with pytest.raises(ValidationError):
        TravelerGroup(adults=2, children=1, children_need_bed=2)


def test_timeline_stay_rejects_invalid_time_order() -> None:
    with pytest.raises(ValidationError):
        TimelineItem(
            sequence=1,
            item_type=TimelineItemType.STAY,
            start_time=time(11, 0),
            end_time=time(10, 0),
            city="Hangzhou",
            stay=StayDetail(
                place_name="West Lake",
                city="Hangzhou",
                purpose=StayPurpose.VISIT,
                category=PlaceCategory.OUTDOOR,
            ),
        )


def test_timeline_item_requires_matching_detail_type() -> None:
    with pytest.raises(ValidationError):
        TimelineItem(
            sequence=1,
            item_type=TimelineItemType.MOVE,
            start_time=time(9, 0),
            end_time=time(10, 0),
            city="Hangzhou",
            stay=StayDetail(place_name="West Lake", city="Hangzhou"),
        )


def test_plan_day_rejects_duplicate_timeline_sequence() -> None:
    item = _visit_item(sequence=1, start=time(9, 0), end=time(10, 0))

    with pytest.raises(ValidationError):
        PlanDay(
            day=1,
            date=date(2026, 7, 1),
            city="Hangzhou",
            timeline=[item, item.model_copy(deep=True)],
        )


def test_plan_day_rejects_overlapping_timeline_items() -> None:
    with pytest.raises(ValidationError):
        PlanDay(
            day=1,
            date=date(2026, 7, 1),
            city="Hangzhou",
            timeline=[
                _visit_item(sequence=1, start=time(9, 0), end=time(11, 0)),
                TimelineItem(
                    sequence=2,
                    item_type=TimelineItemType.MOVE,
                    start_time=time(10, 30),
                    end_time=time(11, 0),
                    city="Hangzhou",
                    move=MoveDetail(
                        origin="West Lake",
                        destination="Hangzhou Museum",
                        mode=TransportMode.TAXI,
                        purpose=MovePurpose.LOCAL,
                    ),
                ),
            ],
        )


def _visit_item(sequence: int, start: time, end: time) -> TimelineItem:
    return TimelineItem(
        sequence=sequence,
        item_type=TimelineItemType.STAY,
        start_time=start,
        end_time=end,
        city="Hangzhou",
        stay=StayDetail(
            place_name="West Lake",
            city="Hangzhou",
            purpose=StayPurpose.VISIT,
            category=PlaceCategory.OUTDOOR,
        ),
    )
