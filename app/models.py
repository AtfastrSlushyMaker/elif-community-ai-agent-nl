from typing import Any, Literal

from pydantic import BaseModel, Field


ActionType = Literal[
    "search_posts",
    "search_comments",
    "get_user_posts",
    "get_user_comments",
    "get_post_by_id",
    "get_related_posts",
    "get_flair_trends",
    "compare_communities",
    "rank_results",
    "summarize_with_citations",
    "extract_actionable_advice",
    "search_flairs",
    "search_rules",
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
    community_id: int | None = None


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
    referenced_comments: list[dict[str, Any]] = Field(default_factory=list)
    referenced_flairs: list[str]
    referenced_rules: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    gaps: list[str] = Field(default_factory=list)
    ranking_factors: list[dict[str, Any]] = Field(default_factory=list)
    next_best_actions: list[str] = Field(default_factory=list)
    model: str
    trace: list[dict[str, Any]] | None = None
