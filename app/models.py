from typing import Any, Literal

from pydantic import BaseModel, Field


ActionType = Literal[
    "search_posts",
    "list_communities",
    "get_community_flairs",
    "get_post_comments",
    "get_trending_posts",
]


class AgentSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    user_id: int | None = None
    act_as_user_id: int | None = None
    include_trace: bool = False
    max_actions: int | None = Field(default=None, ge=1, le=12)


class AgentAction(BaseModel):
    type: ActionType
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class PlannerResponse(BaseModel):
    normalized_query: str
    intent: str = "search"
    actions: list[AgentAction] = Field(default_factory=list)


class AgentSearchResponse(BaseModel):
    query: str
    normalized_query: str
    answer: str
    follow_ups: list[str]
    referenced_posts: list[dict[str, Any]]
    referenced_communities: list[dict[str, Any]]
    referenced_flairs: list[str]
    model: str
    trace: list[dict[str, Any]] | None = None
