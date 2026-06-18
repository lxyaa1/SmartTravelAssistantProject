from __future__ import annotations

from datetime import date as Date
from datetime import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class BudgetLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TransportMode(str, Enum):
    WALK = "walk"
    TAXI = "taxi"
    TRANSIT = "transit"
    TRAIN = "train"
    FLIGHT = "flight"


class PlaceCategory(str, Enum):
    OUTDOOR = "outdoor"
    INDOOR = "indoor"
    FOOD = "food"
    CULTURE = "culture"
    SHOPPING = "shopping"
    HOTEL_AREA = "hotel_area"


class IssueType(str, Enum):
    MISSING_MUST_VISIT = "missing_must_visit"
    BAD_WEATHER = "bad_weather"
    ATTRACTION_CLOSED = "attraction_closed"
    ROUTE_TOO_LONG = "route_too_long"
    DAY_TOO_BUSY = "day_too_busy"
    PREFERENCE_CONFLICT = "preference_conflict"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIME_CONFLICT = "time_conflict"
    NOT_ENOUGH_BEDS = "not_enough_beds"
    TOO_MANY_BEDS_BOOKED = "too_many_beds_booked"
    CHILD_UNFRIENDLY_SCHEDULE = "child_unfriendly_schedule"
    LATE_NIGHT_ACTIVITY_WITH_CHILDREN = "late_night_activity_with_children"
    LONG_TRANSFER_WITH_CHILDREN = "long_transfer_with_children"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentName(str, Enum):
    INPUT_PARSER = "input_parser"
    PREPLAN_QUERY_PLANNER = "preplan_query_planner"
    ROUTE_PLANNER = "route_planner"
    PLAN_CHECK_QUERY_PLANNER = "plan_check_query_planner"
    DATA_COLLECTOR = "data_collector"
    VALIDATOR = "validator"
    REPLANNER = "replanner"
    FINAL_WRITER = "final_writer"


class McpToolName(str, Enum):
    GET_WEATHER = "get_weather"
    GET_ROUTE_TIME = "get_route_time"
    GET_ATTRACTION_DETAIL = "get_attraction_detail"
    SEARCH_ATTRACTIONS = "search_attractions"
    SEARCH_ACCOMMODATION_AREAS = "search_accommodation_areas"


class McpQueryStage(str, Enum):
    PREPLAN = "preplan"
    PLAN_CHECK = "plan_check"


class TravelerGroup(BaseModel):
    adults: int = Field(default=1, ge=1)
    children: int = Field(default=0, ge=0)
    infants: int = Field(default=0, ge=0)
    children_need_bed: int = Field(default=0, ge=0)
    infants_need_bed: int = Field(default=0, ge=0)
    children_ages: list[int] = Field(default_factory=list)
    infants_ages: list[int] = Field(default_factory=list)

    @computed_field
    @property
    def total_people(self) -> int:
        return self.adults + self.children + self.infants

    @computed_field
    @property
    def bed_count(self) -> int:
        return self.adults + self.children_need_bed + self.infants_need_bed

    @computed_field
    @property
    def has_children_or_infants(self) -> bool:
        return self.children > 0 or self.infants > 0

    @model_validator(mode="after")
    def validate_child_counts(self) -> "TravelerGroup":
        if self.children_need_bed > self.children:
            raise ValueError("children_need_bed cannot exceed children")
        if self.infants_need_bed > self.infants:
            raise ValueError("infants_need_bed cannot exceed infants")
        if self.children_ages and len(self.children_ages) != self.children:
            raise ValueError("children_ages length must match children when provided")
        if self.infants_ages and len(self.infants_ages) != self.infants:
            raise ValueError("infants_ages length must match infants when provided")
        if any(age < 2 or age > 17 for age in self.children_ages):
            raise ValueError("children_ages must be between 2 and 17")
        if any(age < 0 or age > 2 for age in self.infants_ages):
            raise ValueError("infants_ages must be between 0 and 2")
        return self


class AccommodationRequirement(BaseModel):
    room_count: int | None = Field(default=None, ge=1)
    bed_count: int = Field(ge=1)
    allow_children_share_bed: bool = True
    prefer_family_room: bool = False
    notes: str = ""


class TripRequest(BaseModel):
    origin: str
    destination: str
    start_date: Date
    end_date: Date
    travelers: TravelerGroup = Field(default_factory=TravelerGroup)
    accommodation: AccommodationRequirement | None = None
    budget_level: BudgetLevel = BudgetLevel.MEDIUM
    preferences: list[str] = Field(default_factory=list)
    must_visit: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)

    @field_validator("travelers", mode="before")
    @classmethod
    def parse_legacy_traveler_count(cls, value: object) -> object:
        if isinstance(value, int):
            return {"adults": value}
        return value

    @model_validator(mode="after")
    def validate_request(self) -> "TripRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.accommodation is None:
            self.accommodation = AccommodationRequirement(
                bed_count=self.travelers.bed_count,
                allow_children_share_bed=self.travelers.children_need_bed == 0
                and self.travelers.infants_need_bed == 0,
                prefer_family_room=self.travelers.has_children_or_infants,
            )
        if self.accommodation.bed_count < self.travelers.bed_count:
            raise ValueError("accommodation.bed_count cannot be less than travelers.bed_count")
        return self


class TransferLeg(BaseModel):
    origin: str
    destination: str
    mode: TransportMode
    estimated_duration_minutes: int = Field(ge=0)
    estimated_distance_km: float = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0, ge=0)
    currency: str = "CNY"
    notes: str = ""


class VisitSlot(BaseModel):
    sequence: int = Field(ge=1)
    place_name: str
    city: str
    category: PlaceCategory
    start_time: time
    end_time: time
    visit_duration_minutes: int = Field(ge=0)
    transport_to_next: TransferLeg | None = None
    estimated_cost: float = Field(default=0, ge=0)
    currency: str = "CNY"
    notes: str = ""

    @model_validator(mode="after")
    def validate_time_order(self) -> "VisitSlot":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        return self


class PlanDay(BaseModel):
    day: int = Field(ge=1)
    date: Date
    city: str
    visits: list[VisitSlot] = Field(default_factory=list)
    accommodation_area: str | None = None
    arrival_transfer: TransferLeg | None = None
    start_transfer_to_first: TransferLeg | None = None
    return_transfer_to_accommodation: TransferLeg | None = None
    total_visit_minutes: int = Field(default=0, ge=0)
    total_transport_minutes: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0, ge=0)
    currency: str = "CNY"
    daily_notes: str = ""

    @model_validator(mode="after")
    def validate_visit_sequence(self) -> "PlanDay":
        sequences = [visit.sequence for visit in self.visits]
        if len(sequences) != len(set(sequences)):
            raise ValueError("visit sequence values must be unique within a day")
        return self


class TripPlan(BaseModel):
    title: str
    origin: str
    destination: str
    days: list[PlanDay] = Field(default_factory=list)
    total_estimated_cost: float = Field(default=0, ge=0)
    currency: str = "CNY"
    assumptions: list[str] = Field(default_factory=list)


class WeatherResult(BaseModel):
    city: str
    date: Date
    condition: str
    warning: str | None = None


class AttractionResult(BaseModel):
    name: str
    city: str
    category: PlaceCategory
    date: Date | None = None
    is_open: bool = True
    opening_hours: str = "09:00-18:00"
    ticket_price: float = Field(default=0, ge=0)
    recommended_duration_minutes: int = Field(default=120, ge=0)
    notes: str = ""


class AccommodationAreaResult(BaseModel):
    area_name: str
    city: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    suitable_for: list[str] = Field(default_factory=list)
    estimated_price_level: BudgetLevel = BudgetLevel.MEDIUM
    notes: str = ""


class RouteResult(BaseModel):
    origin: str
    destination: str
    mode: TransportMode
    duration_minutes: int = Field(ge=0)
    distance_km: float = Field(ge=0)


class McpResults(BaseModel):
    weather: list[WeatherResult] = Field(default_factory=list)
    attractions: list[AttractionResult] = Field(default_factory=list)
    routes: list[RouteResult] = Field(default_factory=list)
    accommodation_areas: list[AccommodationAreaResult] = Field(default_factory=list)


class McpQuery(BaseModel):
    tool_name: McpToolName
    args: dict[str, Any] = Field(default_factory=dict)
    purpose: str
    stage: McpQueryStage


class McpQueryPlan(BaseModel):
    queries: list[McpQuery] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    issue_type: IssueType
    severity: Severity
    day: int | None = None
    date: Date | None = None
    locations: list[str] = Field(default_factory=list)
    reason: str
    suggested_action: str


class FinalPlan(BaseModel):
    content: str
    unresolved_issues: list[ValidationIssue] = Field(default_factory=list)


class InputParserInput(BaseModel):
    raw_user_input: TripRequest


class ParsedRequestOutput(BaseModel):
    agent: Literal[AgentName.INPUT_PARSER] = AgentName.INPUT_PARSER
    request: TripRequest


class PreplanQueryPlannerInput(BaseModel):
    request: TripRequest


class PreplanQueryPlannerOutput(BaseModel):
    agent: Literal[AgentName.PREPLAN_QUERY_PLANNER] = AgentName.PREPLAN_QUERY_PLANNER
    query_plan: McpQueryPlan


class RoutePlannerInput(BaseModel):
    request: TripRequest
    mcp_results: McpResults = Field(default_factory=McpResults)


class RoutePlannerOutput(BaseModel):
    agent: Literal[AgentName.ROUTE_PLANNER] = AgentName.ROUTE_PLANNER
    plan: TripPlan


class PlanCheckQueryPlannerInput(BaseModel):
    plan: TripPlan


class PlanCheckQueryPlannerOutput(BaseModel):
    agent: Literal[AgentName.PLAN_CHECK_QUERY_PLANNER] = AgentName.PLAN_CHECK_QUERY_PLANNER
    query_plan: McpQueryPlan


class DataCollectorInput(BaseModel):
    query_plan: McpQueryPlan
    existing_results: McpResults = Field(default_factory=McpResults)
    default_city: str


class DataCollectorOutput(BaseModel):
    agent: Literal[AgentName.DATA_COLLECTOR] = AgentName.DATA_COLLECTOR
    mcp_results: McpResults


class ValidatorInput(BaseModel):
    request: TripRequest
    plan: TripPlan
    mcp_results: McpResults


class ValidatorOutput(BaseModel):
    agent: Literal[AgentName.VALIDATOR] = AgentName.VALIDATOR
    issues: list[ValidationIssue] = Field(default_factory=list)


class ReplannerInput(BaseModel):
    request: TripRequest
    current_plan: TripPlan
    issues: list[ValidationIssue]
    mcp_results: McpResults
    iteration: int = Field(ge=0)


class ReplannerOutput(BaseModel):
    agent: Literal[AgentName.REPLANNER] = AgentName.REPLANNER
    plan: TripPlan
    addressed_issues: list[ValidationIssue] = Field(default_factory=list)


class FinalWriterInput(BaseModel):
    plan: TripPlan
    unresolved_issues: list[ValidationIssue] = Field(default_factory=list)


class FinalWriterOutput(BaseModel):
    agent: Literal[AgentName.FINAL_WRITER] = AgentName.FINAL_WRITER
    final_plan: FinalPlan
