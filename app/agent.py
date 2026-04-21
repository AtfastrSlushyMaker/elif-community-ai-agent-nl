from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx


import os
from .backend_client import BackendClient
from .models import AgentAction, PlannerResponse
from .prompts import build_planner_prompt, build_query_profile_prompt, build_synthesis_prompt


class CommunitySearchAgent:
    MAX_POSTS = 40
    MAX_COMMUNITIES = 20
    MAX_COMMENTS = 60

    PET_ALIASES: dict[str, set[str]] = {
        "dog": {"dog", "dogs", "puppy", "puppies", "canine"},
        "cat": {"cat", "cats", "kitten", "kittens", "feline"},
        "parrot": {"parrot", "parrots", "macaw", "cockatiel", "cockatoo", "budgie", "budgerigar"},
        "rabbit": {"rabbit", "rabbits", "bunny", "bunnies"},
        "bird": {"bird", "birds", "avian"},
        "hamster": {"hamster", "hamsters"},
        "guinea_pig": {"guinea pig", "guinea pigs", "cavy", "cavies"},
        "fish": {"fish", "fishes", "aquarium"},
    }
    STOP_WORDS: set[str] = {
        "show",
        "me",
        "for",
        "about",
        "from",
        "with",
        "the",
        "and",
        "or",
        "a",
        "an",
        "in",
        "on",
        "of",
        "to",
        "by",
        "posts",
        "post",
        "community",
        "communities",
        "find",
        "search",
        "please",
    }
    MONTHS: dict[str, int] = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    QUARTERS: dict[int, tuple[int, int]] = {
        1: (1, 3),
        2: (4, 6),
        3: (7, 9),
        4: (10, 12),
    }

    def __init__(
        self,
        backend: BackendClient,
        groq_api_key: str,
        groq_model: str,
        max_actions: int,
        groq_api_keys: str = "",
    ) -> None:
        self.backend = backend
        self.groq_api_key = groq_api_key
        self.groq_api_keys = groq_api_keys
        self.groq_model = groq_model
        self.max_actions = max_actions
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self._action_handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
            "search_posts": self._handle_search_posts,
            "search_comments": self._handle_search_comments,
            "get_user_posts": self._handle_get_user_posts,
            "get_user_comments": self._handle_get_user_comments,
            "get_post_by_id": self._handle_get_post_by_id,
            "get_related_posts": self._handle_get_related_posts,
            "get_flair_trends": self._handle_get_flair_trends,
            "compare_communities": self._handle_compare_communities,
            "rank_results": self._handle_rank_results,
            "summarize_with_citations": self._handle_summarize_with_citations,
            "extract_actionable_advice": self._handle_extract_actionable_advice,
            "list_communities": self._handle_list_communities,
            "get_community_flairs": self._handle_get_community_flairs,
            "get_post_comments": self._handle_get_post_comments,
            "get_trending_posts": self._handle_get_trending_posts,
            "search_flairs": self._handle_search_flairs,
            "search_rules": self._handle_search_rules,
        }

    async def run(
        self,
        query: str,
        user_id: int | None = None,
        act_as_user_id: int | None = None,
        max_actions: int | None = None,
        community_id: int | None = None,
    ) -> dict[str, Any]:
        self.backend.reset_cache()
        trace: list[dict[str, Any]] = []
        action_budget = max_actions or self.max_actions
        is_scoped_search = community_id is not None
        query_profile = await self._build_query_profile(query, trace=trace)

        seed_posts = await self.backend.search_posts(
            query,
            user_id=user_id,
            limit=20 if query_profile["recommendation_mode"] else 12,
        )
        if query_profile["author_target"]:
            date_filter = query_profile.get("date_filter")
            author_seed = await self.backend.get_user_posts(
                str(query_profile["author_target"]),
                user_id=user_id,
                limit=16,
                day=self._to_int((date_filter or {}).get("day")),
                month=self._to_int((date_filter or {}).get("month")),
                year=self._to_int((date_filter or {}).get("year")),
            )
            self._merge_posts(seed_posts, self._filter_posts_by_date_filter(author_seed, date_filter))

        communities = await self.backend.list_communities(user_id=user_id)

        if is_scoped_search:
            trace.append({"step": "scope_info", "community_id": community_id, "scope": "single_community"})
            seed_posts = [post for post in seed_posts if self._post_community_id(post) == community_id]
            communities = [community for community in communities if self._to_int(community.get("id")) == community_id]

        seed_posts = self._filter_posts_by_freshness(seed_posts, query_profile["freshness_days"])
        community_index = self._community_index(communities)
        trace.append(
            {
                "step": "seed_fetch",
                "posts": len(seed_posts),
                "communities": len(communities),
                "query_profile": query_profile,
                "impersonation": act_as_user_id is not None,
            }
        )

        plan = await self._plan_actions(
            query=query,
            seed_posts=seed_posts,
            communities=communities,
            action_budget=action_budget,
            trace=trace,
        )
        plan = self._augment_plan_for_profile(plan=plan, query=query, query_profile=query_profile, action_budget=action_budget)
        plan = self._apply_mode_enhancements(plan=plan, query=query, query_profile=query_profile, action_budget=action_budget)

        if is_scoped_search:
            filtered_actions = [action for action in plan.actions if action.type != "compare_communities"]
            trace.append(
                {
                    "step": "plan_optimization",
                    "reason": "scoped_search_context",
                    "actions_before": len(plan.actions),
                    "actions_after": len(filtered_actions),
                }
            )
            plan.actions = filtered_actions

        trace.append({"step": "plan", "normalized_query": plan.normalized_query, "actions": [action.model_dump() for action in plan.actions]})
        community_focus = self._is_community_intent(query=query, intent=plan.intent)

        context: dict[str, Any] = {
            "posts": list(seed_posts),
            "communities": communities,
            "flairs": [],
            "rules": [],
            "comments_by_post": {},
            "matched_comments": [],
            "selected_community_ids": set(),
            "flairs_loaded_for": set(),
            "community_comparisons": [],
            "ranking_factors": [],
            "analysis_notes": [],
            "actionable_advice": [],
            "post_scores": {},
            "community_scores": {},
            "why_post": {},
            "why_community": {},
        }

        if community_focus:
            context["posts"] = []

        for action in plan.actions[:action_budget]:
            started = time.monotonic()
            try:
                result = await self._execute_action(
                    action=action,
                    context=context,
                    user_id=user_id,
                    community_index=community_index,
                    community_focus=community_focus,
                    query_profile=query_profile,
                )
            except Exception as ex:
                trace.append(
                    {
                        "step": "action",
                        "action": action.model_dump(),
                        "result": {"ok": False, "error": str(ex)},
                        "error_type": type(ex).__name__,
                        "duration_ms": round((time.monotonic() - started) * 1000),
                    }
                )
                raise

            trace.append(
                {
                    "step": "action",
                    "action": action.model_dump(),
                    "result": result,
                    "error_type": result.get("error_type"),
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            )

        await self._auto_load_flairs(context=context, user_id=user_id, trace=trace)
        self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
        await self._ensure_comment_context(context=context, user_id=user_id, trace=trace)
        self._refresh_matched_comments_from_query(context=context, query=query)

        if query_profile["recommendation_mode"] or query_profile["explainability_mode"]:
            rank_result = self._rank_results(context=context, query_profile=query_profile, community_index=community_index)
            trace.append({"step": "ranking", "posts_ranked": rank_result["posts_ranked"], "communities_ranked": rank_result["communities_ranked"]})

        synthesis = await self._synthesize(
            query=query,
            normalized_query=plan.normalized_query,
            context=context,
            query_profile=query_profile,
            trace=trace,
        )
        trace.append({"step": "synthesis", "model": synthesis.get("model", self.groq_model)})

        result_type = query_profile.get("result_type", "all")
        referenced_posts = self._trim_posts(context["posts"], context["why_post"])
        referenced_communities = self._trim_communities(context["communities"], context["why_community"])
        referenced_comments = self._trim_comments(context["matched_comments"])
        referenced_flairs = self._trim_flairs(context["flairs"])
        referenced_rules = self._trim_rules(context["rules"])

        if result_type == "posts":
            referenced_communities, referenced_flairs, referenced_rules = [], [], []
        elif result_type == "communities":
            referenced_posts, referenced_comments, referenced_flairs, referenced_rules = [], [], [], []
        elif result_type == "flairs":
            referenced_posts, referenced_communities, referenced_comments, referenced_rules = [], [], [], []
        elif result_type == "rules":
            referenced_posts, referenced_communities, referenced_comments, referenced_flairs = [], [], [], []

        confidence = self._clamp_confidence(
            synthesis.get("confidence"),
            evidence_count=(
                len(referenced_posts)
                + len(referenced_communities)
                + len(referenced_comments)
                + len(referenced_flairs)
                + len(referenced_rules)
            ),
        )
        gaps = self._normalize_gaps(synthesis.get("gaps"), context=context, query_profile=query_profile)
        ranking_factors = context["ranking_factors"][:8]
        next_best_actions = self._next_best_actions(plan=plan, action_budget=action_budget, query_profile=query_profile)

        return {
            "query": query,
            "normalized_query": plan.normalized_query,
            "answer": synthesis["answer"],
            "follow_ups": synthesis["follow_ups"],
            "referenced_posts": referenced_posts,
            "referenced_communities": referenced_communities,
            "referenced_comments": referenced_comments,
            "referenced_flairs": referenced_flairs,
            "referenced_rules": referenced_rules,
            "confidence": confidence,
            "gaps": gaps,
            "ranking_factors": ranking_factors,
            "next_best_actions": next_best_actions,
            "model": synthesis.get("model", self.groq_model),
            "trace": trace,
        }

    async def _plan_actions(
        self,
        query: str,
        seed_posts: list[dict[str, Any]],
        communities: list[dict[str, Any]],
        action_budget: int,
        trace: list[dict[str, Any]],
    ) -> PlannerResponse:
        prompt = build_planner_prompt(query=query, seed_posts=seed_posts, communities=communities, action_budget=action_budget)
        raw = await self._groq_json(prompt, caller="_plan_actions", trace=trace)
        parsed = self._parse_json_candidate(raw)
        if parsed is None:
            trace.append({"step": "llm_parse_failure", "caller": "_plan_actions", "raw_preview": raw[:300]})
        if not isinstance(parsed, dict):
            return self._default_plan(query)

        actions_payload = parsed.get("actions", [])
        actions: list[AgentAction] = []
        if isinstance(actions_payload, list):
            for item in actions_payload:
                if not isinstance(item, dict):
                    continue
                try:
                    actions.append(AgentAction(**item))
                except Exception:
                    continue

        if not actions:
            return self._default_plan(query)

        return PlannerResponse(
            normalized_query=str(parsed.get("normalized_query", query)).strip() or query,
            intent=str(parsed.get("intent", "search")),
            actions=actions[:action_budget],
        )

    def _default_plan(self, query: str) -> PlannerResponse:
        return PlannerResponse(
            normalized_query=query.strip(),
            intent="search",
            actions=[
                AgentAction(type="search_posts", args={"query": query, "limit": 18}, reason="Collect relevant threads"),
                AgentAction(type="list_communities", args={"query": query, "limit": 14}, reason="Find best matching communities"),
                AgentAction(type="search_comments", args={"query": query, "limit_posts": 8, "limit_comments_per_post": 24}, reason="Find direct discussion evidence"),
                AgentAction(type="search_flairs", args={"query": query}, reason="Find relevant post flairs or tags"),
                AgentAction(type="rank_results", args={"strategy": "balanced"}, reason="Rank output with explicit factors"),
                AgentAction(type="extract_actionable_advice", args={}, reason="Derive practical next steps"),
            ],
        )

    async def _execute_action(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        handler = self._action_handlers.get(action.type)
        if handler is None:
            return {"ok": False, "error": f"unsupported action: {action.type}", "error_type": "unsupported_action"}

        try:
            return await handler(action, context, user_id, community_index, community_focus, query_profile)
        except httpx.HTTPStatusError as ex:
            status_code = ex.response.status_code
            if status_code == 404:
                return {"ok": False, "error": "not_found", "error_type": "http_404"}
            if status_code >= 500:
                raise RuntimeError(f"Backend action '{action.type}' failed with HTTP {status_code}") from ex
            return {"ok": False, "error": str(ex), "error_type": f"http_{status_code}"}
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.NetworkError, httpx.TimeoutException) as ex:
            raise RuntimeError(f"Backend action '{action.type}' failed due to connection issue: {ex}") from ex
        except Exception as ex:
            return {"ok": False, "error": str(ex), "error_type": type(ex).__name__}

    async def _handle_search_posts(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        q = str(action.args.get("query", "")).strip()
        if not q:
            return {"ok": False, "error": "missing query", "error_type": "validation"}
        limit = int(action.args.get("limit", 16))
        posts = await self.backend.search_posts(q, user_id=user_id, limit=limit)
        posts = self._filter_posts_by_freshness(posts, query_profile["freshness_days"])
        self._merge_posts(context["posts"], posts)
        self._collect_flairs(context["flairs"], posts)
        return {"ok": True, "posts_added": len(posts)}

    async def _handle_search_comments(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        q = str(action.args.get("query", "")).strip()
        if not q:
            return {"ok": False, "error": "missing query", "error_type": "validation"}
        limit_posts = int(action.args.get("limit_posts", 6))
        limit_comments = int(action.args.get("limit_comments_per_post", 20))
        comments = await self._search_comments(
            query=q,
            posts=context["posts"][: max(1, min(limit_posts, 12))],
            user_id=user_id,
            limit_per_post=limit_comments,
        )
        self._merge_comments(context["matched_comments"], comments)
        return {"ok": True, "comments_found": len(comments)}

    async def _handle_get_user_posts(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        username = str(action.args.get("username", query_profile.get("author_target", ""))).strip()
        if not username:
            return {"ok": False, "error": "missing username", "error_type": "validation"}
        limit = int(action.args.get("limit", 20))
        date_filter = query_profile.get("date_filter")
        posts = await self.backend.get_user_posts(
            username,
            user_id=user_id,
            limit=limit,
            day=self._to_int((date_filter or {}).get("day")),
            month=self._to_int((date_filter or {}).get("month")),
            year=self._to_int((date_filter or {}).get("year")),
        )
        posts = self._filter_posts_by_freshness(posts, query_profile["freshness_days"])
        posts = self._filter_posts_by_date_filter(posts, date_filter)
        self._merge_posts(context["posts"], posts)
        self._collect_flairs(context["flairs"], posts)
        return {"ok": True, "posts_added": len(posts), "username": username}

    async def _handle_get_user_comments(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        username = str(action.args.get("username", query_profile.get("author_target", ""))).strip().lower()
        if not username:
            return {"ok": False, "error": "missing username", "error_type": "validation"}
        await self._ensure_comment_context(context=context, user_id=user_id, trace=[])
        user_comments = []
        for post in context["posts"][:12]:
            post_id = self._to_int(post.get("id"))
            if post_id is None:
                continue
            for comment in self._flatten_comments(context["comments_by_post"].get(str(post_id), []), post_id=post_id):
                if username in str(comment.get("authorName", "")).strip().lower():
                    user_comments.append(comment)
        self._merge_comments(context["matched_comments"], user_comments)
        return {"ok": True, "comments_found": len(user_comments), "username": username}

    async def _handle_get_post_by_id(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        post_id = int(action.args.get("post_id", 0))
        if post_id <= 0:
            return {"ok": False, "error": "missing post_id", "error_type": "validation"}
        post = await self.backend.get_post_by_id(post_id, user_id=user_id)
        if not post:
            return {"ok": True, "post_found": False}
        self._merge_posts(context["posts"], [post])
        self._collect_flairs(context["flairs"], [post])
        return {"ok": True, "post_found": True}

    async def _handle_get_related_posts(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        post_id = int(action.args.get("post_id", 0))
        if post_id <= 0:
            return {"ok": False, "error": "missing post_id", "error_type": "validation"}
        related = await self._get_related_posts(
            post_id=post_id,
            context=context,
            user_id=user_id,
            limit=int(action.args.get("limit", 8)),
        )
        self._merge_posts(context["posts"], related)
        self._collect_flairs(context["flairs"], related)
        return {"ok": True, "related_added": len(related)}

    async def _handle_get_flair_trends(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        trends = self._build_flair_trends(context["posts"])
        for trend in trends:
            context["flairs"].append(trend["name"])
        context["ranking_factors"].append({"factor": "flair_trends", "top": trends[:5]})
        return {"ok": True, "trends_found": len(trends)}

    async def _handle_compare_communities(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        comparison = self._compare_communities(context=context, query_profile=query_profile)
        if comparison:
            context["community_comparisons"].append(comparison)
            context["analysis_notes"].append(f"Comparison: {comparison.get('summary', '')}".strip())
            return {"ok": True, "compared": len(comparison.get("items", []))}
        return {"ok": True, "compared": 0}

    async def _handle_rank_results(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        ranked = self._rank_results(context=context, query_profile=query_profile, community_index=community_index)
        return {"ok": True, **ranked}

    async def _handle_summarize_with_citations(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        note = self._build_cited_summary(context=context)
        if note:
            context["analysis_notes"].append(note)
            return {"ok": True, "summary_added": True}
        return {"ok": True, "summary_added": False}

    async def _handle_extract_actionable_advice(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        advice = self._extract_actionable_advice(context=context)
        if advice:
            context["actionable_advice"].extend(advice)
        return {"ok": True, "advice_items": len(advice)}

    async def _handle_list_communities(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        communities = context["communities"]
        q = str(action.args.get("query", "")).strip().lower()
        limit = int(action.args.get("limit", 12))
        if q:
            ranked = [
                community
                for community in communities
                if q in str(community.get("name", "")).lower() or q in str(community.get("description", "")).lower()
            ]
            context["communities"] = ranked[: max(1, min(limit, 30))]
        else:
            context["communities"] = communities[: max(1, min(limit, 30))]
        context["selected_community_ids"] = {
            cid for cid in (self._to_int(community.get("id")) for community in context["communities"]) if cid is not None
        }
        if community_focus:
            self._filter_posts_by_selected_communities(context["posts"], context["selected_community_ids"])
        return {"ok": True, "communities_selected": len(context["communities"])}

    async def _handle_get_community_flairs(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        community_id = int(action.args.get("community_id", 0))
        if community_id <= 0:
            return {"ok": False, "error": "missing community_id", "error_type": "validation"}
        flairs = await self.backend.list_flairs(community_id)
        names = [str(item.get("name", "")).strip() for item in flairs if item.get("name")]
        context["flairs"].extend(names)
        context["flairs_loaded_for"].add(community_id)
        if community_id in community_index:
            community_index[community_id]["_flairs"] = names
        return {"ok": True, "flairs_added": len(names)}

    async def _handle_get_post_comments(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        post_id = int(action.args.get("post_id", 0))
        if post_id <= 0:
            return {"ok": False, "error": "missing post_id", "error_type": "validation"}
        comments = await self.backend.get_post_comments(post_id, user_id=user_id)
        context["comments_by_post"][str(post_id)] = comments[: self.MAX_COMMENTS]
        self._refresh_matched_comments_from_query(context=context, query=str(action.args.get("query", "")))
        return {"ok": True, "comments_loaded": len(comments)}

    async def _handle_get_trending_posts(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        sort = str(action.args.get("sort", "HOT")).upper()
        window = str(action.args.get("window", "ALL")).upper()
        limit = int(action.args.get("limit", 8))
        posts = await self.backend.get_trending_posts(user_id=user_id, sort=sort, window=window, limit=limit)
        posts = self._filter_posts_by_freshness(posts, query_profile["freshness_days"])
        self._merge_posts(context["posts"], posts)
        self._collect_flairs(context["flairs"], posts)
        return {"ok": True, "trending_added": len(posts)}

    async def _handle_search_flairs(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        q = str(action.args.get("query", "")).strip()
        if not q:
            return {"ok": False, "error": "missing query", "error_type": "validation"}
        flairs = await self.backend.search_flairs_by_name(q, context["communities"])
        if flairs:
            context["flairs"].extend([flair.get("name") for flair in flairs if flair.get("name")])
        return {"ok": True, "flairs_found": len(flairs)}

    async def _handle_search_rules(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
        q = str(action.args.get("query", "")).strip()
        if not q:
            return {"ok": False, "error": "missing query", "error_type": "validation"}
        rules = await self.backend.search_rules_by_content(q, context["communities"])
        if rules:
            context["rules"].extend(rules)
        return {"ok": True, "rules_found": len(rules)}

    async def _synthesize(
        self,
        query: str,
        normalized_query: str,
        context: dict[str, Any],
        query_profile: dict[str, Any],
        trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt_context = {
            "trimmed_posts": self._trim_posts(context["posts"], context["why_post"]),
            "trimmed_communities": self._trim_communities(context["communities"], context["why_community"]),
            "trimmed_comments": self._trim_comments(context["matched_comments"]),
            "trimmed_flairs": self._trim_flairs(context["flairs"]),
            "trimmed_rules": self._trim_rules(context["rules"]),
            "ranking_factors": context["ranking_factors"],
            "analysis_notes": context["analysis_notes"],
            "actionable_advice": context["actionable_advice"],
            "user_context": self._collect_user_context(context["posts"], context["comments_by_post"]),
            "comments_by_post": context["comments_by_post"],
        }
        prompt = build_synthesis_prompt(query=query, normalized_query=normalized_query, context=prompt_context, query_profile=query_profile)
        raw = await self._groq_json(prompt, caller="_synthesize", trace=trace)
        parsed = self._parse_json_candidate(raw)
        if parsed is None:
            trace.append({"step": "llm_parse_failure", "caller": "_synthesize", "raw_preview": raw[:300]})
        if not isinstance(parsed, dict):
            return {
                "answer": raw.strip() or "I could not build a reliable answer from the available context.",
                "follow_ups": [
                    f"Show me top posts about {normalized_query}",
                    f"Compare two communities related to {normalized_query}",
                    f"What should I do first about {normalized_query}?",
                ],
                "confidence": 0.35,
                "gaps": ["Need stronger direct evidence from comments and recent posts."],
                "model": self.groq_model,
            }

        answer = str(parsed.get("answer", "")).strip() or "I could not build a reliable answer from the available context."
        follow_ups = self._normalize_follow_ups(parsed.get("follow_ups", []), normalized_query=normalized_query)
        confidence = self._clamp_confidence(parsed.get("confidence"), evidence_count=0)
        gaps = self._normalize_gaps(parsed.get("gaps"), context=context, query_profile=query_profile)
        return {"answer": answer, "follow_ups": follow_ups, "confidence": confidence, "gaps": gaps, "model": self.groq_model}

    async def _groq_json(self, prompt: str, caller: str, trace: list[dict[str, Any]] | None = None) -> str:
        import asyncio
        prompt_chars = len(prompt)
        if trace is not None:
            trace.append({"step": "prompt_size", "caller": caller, "chars": prompt_chars})
            if prompt_chars > 12000:
                trace.append({"step": "prompt_size_warning", "caller": caller, "chars": prompt_chars})

        api_keys = self.groq_api_keys
        if api_keys:
            key_list = [k.strip() for k in api_keys.split(",") if k.strip()]
        else:
            key_list = [self.groq_api_key]

        payload = {
            "model": self.groq_model,
            "temperature": 0.2,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        retryable_statuses = {429, 503}

        async def try_key(api_key, key_idx):
            import traceback
            async with httpx.AsyncClient(timeout=httpx.Timeout(25)) as client:
                for attempt in range(3):
                    try:
                        print(f"[Groq] Trying key {key_idx+1}/{len(key_list)} (attempt {attempt+1})")
                        if trace is not None:
                            trace.append({"step": "groq_try_key", "caller": caller, "api_key": key_idx, "attempt": attempt + 1})
                        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                        response = await client.post(self.groq_url, headers=headers, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        print(f"[Groq] Key {key_idx+1} succeeded.")
                        return str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
                    except httpx.TimeoutException as ex:
                        print(f"[Groq] Timeout for key {key_idx+1} on attempt {attempt+1}: {ex}")
                        if attempt < 2:
                            if trace is not None:
                                trace.append({"step": "groq_retry", "caller": caller, "api_key": key_idx, "attempt": attempt + 1, "error_type": type(ex).__name__})
                            await asyncio.sleep(2**attempt)
                            continue
                        break
                    except httpx.HTTPStatusError as ex:
                        status = ex.response.status_code
                        print(f"[Groq] HTTP error for key {key_idx+1} on attempt {attempt+1}: {status} - {ex}")
                        if status == 429:
                            if trace is not None:
                                trace.append({"step": "groq_rate_limited", "caller": caller, "api_key": key_idx, "attempt": attempt + 1, "error_type": f"http_{status}"})
                            print(f"[Groq] Key {key_idx+1} rate-limited, rotating to next key...")
                            await asyncio.sleep(1)
                            break
                        if status in retryable_statuses and attempt < 2:
                            if trace is not None:
                                trace.append({"step": "groq_retry", "caller": caller, "api_key": key_idx, "attempt": attempt + 1, "error_type": f"http_{status}"})
                            await asyncio.sleep(2**attempt)
                            continue
                        if trace is not None:
                            trace.append({"step": "groq_http_error_next_key", "caller": caller, "api_key": key_idx, "status": status})
                        break
                    except Exception as ex:
                        print(f"[Groq] Exception for key {key_idx+1} on attempt {attempt+1}: {ex}\n{traceback.format_exc()}")
                        if trace is not None:
                            trace.append({"step": "groq_exception_next_key", "caller": caller, "api_key": key_idx, "error": str(ex)})
                        break
            print(f"[Groq] Key {key_idx+1} failed all attempts.")
            return None

        for idx, api_key in enumerate(key_list):
            result = await try_key(api_key, idx)
            if result:
                return result
        if trace is not None:
            trace.append({"step": "groq_all_keys_failed", "caller": caller})
        raise RuntimeError("All GROQ API keys failed or were rate-limited.")



    def _parse_json_candidate(self, raw: str) -> dict[str, Any] | None:
        source = (raw or "").strip().replace("```json", "").replace("```", "").strip()
        if not source:
            return None
        try:
            parsed = json.loads(source)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        start = source.find("{")
        end = source.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(source[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def _merge_posts(self, base: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> None:
        seen = {self._to_int(post.get("id")) for post in base}
        for post in incoming:
            post_id = self._to_int(post.get("id"))
            if post_id in seen:
                continue
            base.append(post)
            seen.add(post_id)
        base.sort(key=lambda post: (int(post.get("voteScore", 0)), int(post.get("commentCount", 0))), reverse=True)
        del base[self.MAX_POSTS :]

    def _merge_comments(self, base: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> None:
        seen = {(self._to_int(comment.get("postId")), self._to_int(comment.get("id"))) for comment in base}
        for comment in incoming:
            key = (self._to_int(comment.get("postId")), self._to_int(comment.get("id")))
            if key in seen:
                continue
            base.append(comment)
            seen.add(key)
        base.sort(key=lambda comment: int(comment.get("score", 0)), reverse=True)
        del base[self.MAX_COMMENTS :]

    def _collect_flairs(self, flair_list: list[str], posts: list[dict[str, Any]]) -> None:
        for post in posts:
            flair = str(post.get("flairName", "")).strip()
            if flair:
                flair_list.append(flair)

    def _trim_posts(self, posts: list[dict[str, Any]], why_map: dict[int, str] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for post in posts[: self.MAX_POSTS]:
            post_id = self._to_int(post.get("id")) or 0
            out.append(
                {
                    "id": post.get("id"),
                    "communityId": self._post_community_id(post),
                    "communitySlug": post.get("communitySlug"),
                    "userId": post.get("userId"),
                    "authorName": post.get("authorName"),
                    "title": post.get("title"),
                    "content": str(post.get("content", ""))[:360],
                    "flairName": post.get("flairName"),
                    "type": post.get("type"),
                    "voteScore": post.get("voteScore"),
                    "viewCount": post.get("viewCount"),
                    "commentCount": post.get("commentCount"),
                    "createdAt": post.get("createdAt"),
                    "why_selected": (why_map or {}).get(post_id),
                }
            )
        return out

    def _trim_communities(self, communities: list[dict[str, Any]], why_map: dict[int, str] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for community in communities[: self.MAX_COMMUNITIES]:
            community_id = self._to_int(community.get("id")) or 0
            out.append(
                {
                    "id": community.get("id"),
                    "name": community.get("name"),
                    "slug": community.get("slug"),
                    "description": str(community.get("description", ""))[:220],
                    "memberCount": community.get("memberCount"),
                    "type": community.get("type"),
                    "bannerUrl": community.get("bannerUrl"),
                    "iconUrl": community.get("iconUrl"),
                    "createdAt": community.get("createdAt"),
                    "userRole": community.get("userRole"),
                    "why_selected": (why_map or {}).get(community_id),
                }
            )
        return out

    def _trim_comments(self, comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for comment in comments[: self.MAX_COMMENTS]:
            out.append(
                {
                    "id": comment.get("id"),
                    "postId": comment.get("postId"),
                    "authorName": comment.get("authorName"),
                    "content": str(comment.get("content", ""))[:260],
                    "createdAt": comment.get("createdAt"),
                    "score": comment.get("score", 0),
                }
            )
        return out

    def _trim_flairs(self, flairs: list[Any]) -> list[str]:
        return sorted({str(item.get("name", "") if isinstance(item, dict) else item).strip() for item in flairs if item})[:30]

    def _trim_rules(self, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[int | None, int | None]] = set()
        for rule in rules:
            key = (self._to_int(rule.get("community_id")), self._to_int(rule.get("id")))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "id": rule.get("id"),
                    "title": rule.get("title"),
                    "description": str(rule.get("description", ""))[:260],
                    "community_id": rule.get("community_id"),
                    "community_name": rule.get("community_name"),
                    "order": rule.get("order"),
                }
            )
            if len(out) == self.MAX_COMMUNITIES:
                break
        return out

    def _community_index(self, communities: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        index: dict[int, dict[str, Any]] = {}
        for community in communities:
            community_id = self._to_int(community.get("id"))
            if community_id is not None:
                index[community_id] = community
        return index

    def _is_community_intent(self, query: str, intent: str | None) -> bool:
        source = f"{query} {intent or ''}".lower()
        return any(
            phrase in source
            for phrase in (
                "community",
                "communities",
                "group",
                "groups",
                "where should i join",
                "which community",
                "find communities",
                "show me communities",
            )
        )

    def _filter_posts_by_selected_communities(self, posts: list[dict[str, Any]], selected_ids: set[int]) -> None:
        if not selected_ids:
            posts.clear()
            return
        posts[:] = [post for post in posts if self._post_community_id(post) in selected_ids]

    def _post_community_id(self, post: dict[str, Any]) -> int | None:
        return self._to_int(post.get("communityId")) or self._to_int(post.get("community_id"))

    def _to_int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _fallback_flairs_from_posts(self, community_id: int, user_id: int | None) -> list[str]:
        try:
            posts = await self.backend.get_community_posts(
                community_id=community_id,
                user_id=user_id,
                sort="HOT",
                window="ALL",
                limit=self.MAX_COMMUNITIES,
            )
        except Exception:
            return []
        flairs: list[str] = []
        for post in posts:
            flair = str(post.get("flairName", "")).strip()
            if flair:
                flairs.append(flair)
        return flairs

    async def _auto_load_flairs(self, context: dict[str, Any], user_id: int | None, trace: list[dict[str, Any]]) -> None:
        selected_community_ids = [community_id for community_id in list(context["selected_community_ids"])[:3] if community_id not in context["flairs_loaded_for"]]
        semaphore = asyncio.Semaphore(5)

        async def load_one(community_id: int) -> dict[str, Any]:
            try:
                async with semaphore:
                    flairs = await self.backend.list_flairs(community_id)
                names = [str(item.get("name", "")).strip() for item in flairs if item.get("name")]
                return {"community_id": community_id, "names": names}
            except Exception as ex:
                fallback_names = await self._fallback_flairs_from_posts(community_id=community_id, user_id=user_id)
                return {"community_id": community_id, "names": fallback_names, "error": str(ex), "fallback": True}

        for outcome in await asyncio.gather(*(load_one(community_id) for community_id in selected_community_ids), return_exceptions=True):
            if isinstance(outcome, Exception):
                trace.append({"step": "auto_flairs", "error": str(outcome), "error_type": type(outcome).__name__})
                continue
            community_id = outcome["community_id"]
            names = outcome.get("names", [])
            context["flairs"].extend(names)
            if names:
                context["flairs_loaded_for"].add(community_id)
            trace.append(
                {
                    "step": "auto_flairs",
                    "community_id": community_id,
                    "flairs_added": len(names),
                    "error": outcome.get("error"),
                    "fallback_flairs_added": len(names) if outcome.get("fallback") else 0,
                }
            )

    async def _ensure_comment_context(self, context: dict[str, Any], user_id: int | None, trace: list[dict[str, Any]]) -> None:
        top_posts = sorted(
            context["posts"],
            key=lambda post: (int(post.get("voteScore", 0)), int(post.get("commentCount", 0))),
            reverse=True,
        )[:6]
        semaphore = asyncio.Semaphore(5)

        async def load_comments(post: dict[str, Any]) -> dict[str, Any]:
            post_id = self._to_int(post.get("id"))
            if post_id is None:
                return {"skip": True}
            key = str(post_id)
            if key in context["comments_by_post"]:
                return {"skip": True, "post_id": post_id}
            try:
                async with semaphore:
                    comments = await self.backend.get_post_comments(post_id, user_id=user_id)
                return {"post_id": post_id, "comments": comments}
            except Exception as ex:
                return {"post_id": post_id, "error": str(ex), "error_type": type(ex).__name__}

        for outcome in await asyncio.gather(*(load_comments(post) for post in top_posts), return_exceptions=True):
            if isinstance(outcome, Exception):
                trace.append({"step": "auto_comments", "error": str(outcome), "error_type": type(outcome).__name__})
                continue
            if outcome.get("skip"):
                continue
            post_id = outcome["post_id"]
            if "comments" in outcome:
                comments = outcome["comments"]
                context["comments_by_post"][str(post_id)] = comments[: self.MAX_COMMENTS]
                trace.append({"step": "auto_comments", "post_id": post_id, "comments_loaded": len(comments)})
            else:
                trace.append(
                    {
                        "step": "auto_comments",
                        "post_id": post_id,
                        "error": outcome.get("error"),
                        "error_type": outcome.get("error_type"),
                    }
                )

    async def _build_query_profile(self, query: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
        heuristic_profile = self._build_heuristic_query_profile(query)
        llm_profile = await self._interpret_query_with_llm(query, trace=trace)
        merged_profile = self._merge_query_profiles(heuristic_profile, llm_profile or {})
        merged_profile.setdefault("date_filter", heuristic_profile.get("date_filter"))
        return merged_profile

    def _build_heuristic_query_profile(self, query: str) -> dict[str, Any]:
        lowered = (query or "").strip().lower()
        author_target = self._extract_author_target(lowered)
        animals = sorted(self._extract_animals(lowered))
        topic_tokens = [token for token in self._tokenize(lowered) if token not in self.STOP_WORDS]
        result_type = self._infer_result_type(lowered)
        freshness_days = self._extract_freshness_days(lowered)
        comparison_mode = any(word in lowered for word in ("compare", "vs", "versus", "difference"))
        recommendation_mode = any(word in lowered for word in ("recommend", "best", "top", "rank"))
        explainability_mode = recommendation_mode or any(word in lowered for word in ("why", "reason", "explain"))

        return {
            "author_target": author_target,
            "strict_author": author_target is not None,
            "animals": animals,
            "strict_animals": bool(animals),
            "topic_tokens": topic_tokens[:10],
            "result_type": result_type,
            "freshness_days": freshness_days,
            "comparison_mode": comparison_mode,
            "recommendation_mode": recommendation_mode,
            "explainability_mode": explainability_mode,
            "multi_hop_mode": True,
            "date_filter": self._extract_date_filter(query),
        }

    async def _interpret_query_with_llm(self, query: str, trace: list[dict[str, Any]]) -> dict[str, Any] | None:
        prompt = build_query_profile_prompt(query)
        try:
            raw = await self._groq_json(prompt, caller="_interpret_query_with_llm", trace=trace)
            parsed = self._parse_json_candidate(raw)
            if parsed is None:
                trace.append({"step": "llm_parse_failure", "caller": "_interpret_query_with_llm", "raw_preview": raw[:300]})
            if not isinstance(parsed, dict):
                return None

            author_target = parsed.get("author_target")
            if author_target:
                author_target = str(author_target).strip().lower() or None

            animals_raw = parsed.get("animals", [])
            animals = sorted(
                str(animal).strip().lower()
                for animal in (animals_raw if isinstance(animals_raw, list) else [])
                if str(animal).strip()
            )

            search_query = str(parsed.get("search_query", "")).strip() or None
            result_type = str(parsed.get("result_type", "all")).strip().lower()
            if result_type not in ("posts", "communities", "flairs", "rules", "all"):
                result_type = "all"
            freshness_days = self._coerce_days(parsed.get("freshness_days"))
            recommendation_confidence = self._coerce_confidence(parsed.get("recommendation_confidence"))
            comparison_confidence = self._coerce_confidence(parsed.get("comparison_confidence"))
            recommendation_mode = recommendation_confidence >= 0.65 and bool(parsed.get("recommendation_mode", False))
            comparison_mode = comparison_confidence >= 0.65 and bool(parsed.get("comparison_mode", False))
            explainability_mode = bool(parsed.get("explainability_mode", False)) or recommendation_mode
            is_author = bool(parsed.get("is_author_query", False))
            is_animal = bool(parsed.get("is_animal_query", False))

            return {
                "author_target": author_target,
                "strict_author": is_author and author_target is not None,
                "animals": animals,
                "strict_animals": is_animal and bool(animals),
                "topic_tokens": self._tokenize(search_query or "")[:10] if search_query else [],
                "result_type": result_type,
                "freshness_days": freshness_days,
                "comparison_mode": comparison_mode,
                "recommendation_mode": recommendation_mode,
                "recommendation_confidence": recommendation_confidence,
                "comparison_confidence": comparison_confidence,
                "explainability_mode": explainability_mode,
                "multi_hop_mode": True,
                "llm_interpreted": True,
                "date_filter": self._extract_date_filter(query),
            }
        except Exception:
            return None

    def _merge_query_profiles(self, heuristic_profile: dict[str, Any], llm_profile: dict[str, Any]) -> dict[str, Any]:
        merged = dict(heuristic_profile)
        for key, value in llm_profile.items():
            if not self._is_empty_profile_value(value):
                merged[key] = value
        for key, heuristic_value in heuristic_profile.items():
            if key not in merged or self._is_empty_profile_value(merged.get(key)):
                merged[key] = heuristic_value
        return merged

    def _is_empty_profile_value(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict, set, tuple)):
            return len(value) == 0
        return False

    def _extract_date_filter(self, query: str) -> dict[str, Any] | None:
        lowered = (query or "").strip().lower()
        if not lowered:
            return None

        quarter_match = re.search(r"\bq([1-4])\s+(\d{4})\b", lowered)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = int(quarter_match.group(2))
            month_start, month_end = self.QUARTERS[quarter]
            return {"day": None, "month": month_start, "year": year, "month_range": [month_start, month_end]}

        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered)
        if iso_match:
            year, month, day = map(int, iso_match.groups())
            return self._validated_date_filter(day=day, month=month, year=year)

        slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", lowered)
        if slash_match:
            day, month, year = map(int, slash_match.groups())
            return self._validated_date_filter(day=day, month=month, year=year)

        month_names = "|".join(self.MONTHS.keys())
        month_first = re.search(rf"\b({month_names})\s+(\d{{1,2}})(?:\s+(\d{{4}}))?\b", lowered)
        if month_first:
            month = self.MONTHS[month_first.group(1)]
            day = int(month_first.group(2))
            year = int(month_first.group(3)) if month_first.group(3) else datetime.now(UTC).year
            return self._validated_date_filter(day=day, month=month, year=year)

        day_first = re.search(rf"\b(\d{{1,2}})\s+({month_names})(?:\s+(\d{{4}}))?\b", lowered)
        if day_first:
            day = int(day_first.group(1))
            month = self.MONTHS[day_first.group(2)]
            year = int(day_first.group(3)) if day_first.group(3) else datetime.now(UTC).year
            return self._validated_date_filter(day=day, month=month, year=year)

        month_only = re.search(rf"\b({month_names})(?:\s+(\d{{4}}))?\b", lowered)
        if month_only:
            month = self.MONTHS[month_only.group(1)]
            year = int(month_only.group(2)) if month_only.group(2) else datetime.now(UTC).year
            return self._validated_date_filter(day=None, month=month, year=year)

        return None

    def _validated_date_filter(self, day: int | None, month: int | None, year: int | None) -> dict[str, Any] | None:
        if year is None and month is None and day is None:
            return None
        try:
            if year is not None and month is not None and day is not None:
                datetime(year, month, day, tzinfo=UTC)
            elif year is not None and month is not None:
                datetime(year, month, 1, tzinfo=UTC)
        except ValueError:
            return None
        return {"day": day, "month": month, "year": year}

    def _filter_posts_by_date_filter(self, posts: list[dict[str, Any]], date_filter: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not date_filter:
            return posts
        day = self._to_int(date_filter.get("day"))
        month = self._to_int(date_filter.get("month"))
        year = self._to_int(date_filter.get("year"))
        month_range = date_filter.get("month_range")

        filtered: list[dict[str, Any]] = []
        for post in posts:
            created = self._parse_dt(post.get("createdAt"))
            if created is None:
                continue
            if year is not None and created.year != year:
                continue
            if isinstance(month_range, list) and len(month_range) == 2:
                if not (int(month_range[0]) <= created.month <= int(month_range[1])):
                    continue
            elif month is not None and created.month != month:
                continue
            if day is not None and created.day != day:
                continue
            filtered.append(post)
        return filtered

    def _augment_plan_for_profile(
        self,
        plan: PlannerResponse,
        query: str,
        query_profile: dict[str, Any],
        action_budget: int,
    ) -> PlannerResponse:
        actions = list(plan.actions)
        if query_profile["author_target"]:
            author_q = str(query_profile["author_target"])
            if not any(action.type == "get_user_posts" for action in actions):
                actions.insert(0, AgentAction(type="get_user_posts", args={"username": author_q, "limit": 18}, reason="Collect author-specific posts"))
            if not any(action.type == "get_user_comments" for action in actions):
                actions.append(AgentAction(type="get_user_comments", args={"username": author_q}, reason="Collect author-specific comment evidence"))

        if query_profile["animals"]:
            animal_q = str(query_profile["animals"][0])
            if not any(action.type == "search_posts" and animal_q in str(action.args.get("query", "")).lower() for action in actions):
                actions.append(AgentAction(type="search_posts", args={"query": animal_q, "limit": 14}, reason="Fetch species-focused context"))

        if query_profile["freshness_days"] is not None and not any(action.type == "rank_results" for action in actions):
            actions.append(AgentAction(type="rank_results", args={"strategy": "freshness_weighted"}, reason="Enforce freshness-aware ranking"))

        if not actions:
            actions = self._default_plan(query).actions
        return PlannerResponse(normalized_query=plan.normalized_query, intent=plan.intent, actions=actions[:action_budget])

    def _apply_mode_enhancements(
        self,
        plan: PlannerResponse,
        query: str,
        query_profile: dict[str, Any],
        action_budget: int,
    ) -> PlannerResponse:
        actions = list(plan.actions)
        if query_profile["multi_hop_mode"] and not any(action.type == "search_comments" for action in actions):
            actions.append(AgentAction(type="search_comments", args={"query": query, "limit_posts": 6}, reason="Add discussion depth for multi-hop reasoning"))
        if query_profile["comparison_mode"] and not any(action.type == "compare_communities" for action in actions):
            actions.append(AgentAction(type="compare_communities", args={}, reason="Generate side-by-side community comparison"))
        if query_profile["recommendation_mode"] and not any(action.type == "rank_results" for action in actions):
            actions.append(AgentAction(type="rank_results", args={"strategy": "balanced"}, reason="Rank best options for recommendation"))
        if not any(action.type == "summarize_with_citations" for action in actions):
            actions.append(AgentAction(type="summarize_with_citations", args={}, reason="Provide citation-grounded summary notes"))
        if not any(action.type == "extract_actionable_advice" for action in actions):
            actions.append(AgentAction(type="extract_actionable_advice", args={}, reason="Extract practical next steps from evidence"))
        return PlannerResponse(normalized_query=plan.normalized_query, intent=plan.intent, actions=actions[:action_budget])

    def _extract_author_target(self, lowered_query: str) -> str | None:
        patterns = (
            r"(?:from|by)\s+user\s+['\"]?([a-z0-9._\-\s]{2,80})['\"]?",
            r"(?:from|by)\s+@?([a-z0-9._\-]{2,80})",
            r"posts?\s+of\s+@?([a-z0-9._\-]{2,80})",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered_query)
            if not match:
                continue
            candidate = self._normalize_author_candidate(match.group(1) or "")
            if candidate:
                return candidate
        return None

    def _normalize_author_candidate(self, raw: str) -> str:
        candidate = (raw or "").strip(" '\"")
        candidate = re.split(r"\b(?:for|about|in|with|on)\b", candidate, maxsplit=1)[0].strip()
        return candidate

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) >= 2]

    def _extract_animals(self, lowered_query: str) -> set[str]:
        animals: set[str] = set()
        for canonical, aliases in self.PET_ALIASES.items():
            for alias in aliases:
                if self._contains_alias(lowered_query, alias):
                    animals.add(canonical)
                    break
        return animals

    def _contains_alias(self, text: str, alias: str) -> bool:
        normalized_alias = re.sub(r"\s+", r"\\s+", re.escape(alias.strip().lower()))
        return re.search(rf"\b{normalized_alias}\b", text.lower()) is not None

    def _post_animals(self, post: dict[str, Any], community_index: dict[int, dict[str, Any]]) -> set[str]:
        community = community_index.get(self._post_community_id(post) or -1, {})
        haystack = " ".join(
            [
                str(post.get("title", "")),
                str(post.get("content", "")),
                str(post.get("flairName", "")),
                str(post.get("communitySlug", "")),
                str(community.get("name", "")),
                str(community.get("description", "")),
            ]
        ).lower()
        return self._extract_animals(haystack)

    def _apply_query_constraints(
        self,
        context: dict[str, Any],
        query_profile: dict[str, Any],
        community_index: dict[int, dict[str, Any]],
    ) -> None:
        requested_animals = set(query_profile["animals"])
        if requested_animals and context["communities"]:
            scoped_communities = []
            for community in context["communities"]:
                haystack = " ".join(
                    [str(community.get("name", "")), str(community.get("slug", "")), str(community.get("description", ""))]
                ).lower()
                if requested_animals.intersection(self._extract_animals(haystack)):
                    scoped_communities.append(community)
            if scoped_communities:
                context["communities"] = scoped_communities[: self.MAX_COMMUNITIES]

        context["posts"] = self._filter_posts_by_freshness(context["posts"], query_profile.get("freshness_days"))
        context["posts"] = self._filter_posts_by_date_filter(context["posts"], query_profile.get("date_filter"))
        posts = context["posts"]
        if not posts:
            return

        author_target = query_profile["author_target"]
        topic_tokens = list(query_profile["topic_tokens"])
        scored: list[tuple[int, dict[str, Any]]] = []
        for post in posts:
            score = int(post.get("voteScore", 0)) + int(post.get("commentCount", 0)) * 2
            text = " ".join(
                [
                    str(post.get("title", "")),
                    str(post.get("content", "")),
                    str(post.get("flairName", "")),
                    str(post.get("communitySlug", "")),
                ]
            ).lower()
            post_author = str(post.get("authorName", "")).strip().lower()
            post_animals = self._post_animals(post, community_index)

            if author_target:
                score += 120 if author_target in post_author else -100
            for token in topic_tokens:
                if token in text:
                    score += 8
            if requested_animals:
                overlap = requested_animals.intersection(post_animals)
                if overlap:
                    score += 70
                elif post_animals:
                    score -= 80
                else:
                    score -= 30
            scored.append((score, post))

        scored.sort(key=lambda item: item[0], reverse=True)
        filtered = [post for score, post in scored if score > 0]
        if query_profile["strict_author"] and author_target:
            author_only = [post for post in filtered if author_target in str(post.get("authorName", "")).lower()]
            if author_only:
                filtered = author_only
        if query_profile["strict_animals"] and requested_animals:
            animal_only = [post for post in filtered if requested_animals.intersection(self._post_animals(post, community_index))]
            if animal_only:
                filtered = animal_only
        context["posts"] = filtered[: self.MAX_POSTS]

    async def _search_comments(
        self,
        query: str,
        posts: list[dict[str, Any]],
        user_id: int | None,
        limit_per_post: int = 20,
    ) -> list[dict[str, Any]]:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []
        results: list[dict[str, Any]] = []
        for post in posts:
            post_id = self._to_int(post.get("id"))
            if post_id is None:
                continue
            comments = await self.backend.get_post_comments(post_id, user_id=user_id)
            for flat in self._flatten_comments(comments, post_id=post_id):
                content = str(flat.get("content", "")).lower()
                overlap = sum(1 for token in query_tokens if token in content)
                if overlap > 0:
                    flat["score"] = overlap
                    results.append(flat)
            if limit_per_post > 0:
                results = sorted(results, key=lambda comment: int(comment.get("score", 0)), reverse=True)[: max(1, limit_per_post * len(posts))]
        return results

    def _flatten_comments(self, comments: Any, post_id: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def walk(nodes: Any) -> None:
            if not isinstance(nodes, list):
                return
            for comment in nodes:
                if not isinstance(comment, dict):
                    continue
                out.append(
                    {
                        "id": comment.get("id"),
                        "postId": post_id,
                        "authorName": comment.get("authorName"),
                        "content": comment.get("content"),
                        "createdAt": comment.get("createdAt"),
                        "score": comment.get("voteScore", 0),
                    }
                )
                walk(comment.get("replies"))

        walk(comments)
        return out

    async def _get_related_posts(self, post_id: int, context: dict[str, Any], user_id: int | None, limit: int) -> list[dict[str, Any]]:
        target = await self.backend.get_post_by_id(post_id, user_id=user_id)
        if not target:
            return []
        related: list[dict[str, Any]] = []
        title_tokens = [token for token in self._tokenize(str(target.get("title", ""))) if token not in self.STOP_WORDS][:4]
        flair = str(target.get("flairName", "")).strip()
        community_id = self._post_community_id(target)

        for token in title_tokens[:2]:
            posts = await self.backend.search_posts(token, user_id=user_id, limit=10)
            related.extend(posts)
        if flair:
            related.extend(await self.backend.search_posts(flair, user_id=user_id, limit=10))
        if community_id is not None:
            related.extend(await self.backend.get_community_posts(community_id=community_id, user_id=user_id, limit=12))

        dedup: list[dict[str, Any]] = []
        seen: set[int] = {post_id}
        for post in related:
            candidate_id = self._to_int(post.get("id"))
            if candidate_id is None or candidate_id in seen:
                continue
            seen.add(candidate_id)
            dedup.append(post)
            if len(dedup) >= max(1, min(limit, self.MAX_COMMUNITIES)):
                break
        return dedup

    def _build_flair_trends(self, posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trends: dict[str, dict[str, Any]] = {}
        now = datetime.now(UTC)
        for post in posts:
            flair = str(post.get("flairName", "")).strip()
            if not flair:
                continue
            entry = trends.setdefault(flair, {"name": flair, "count": 0, "recent": 0})
            entry["count"] += 1
            created = self._parse_dt(post.get("createdAt"))
            if created and (now - created) <= timedelta(days=7):
                entry["recent"] += 1
        ranked = sorted(trends.values(), key=lambda trend: (int(trend["recent"]), int(trend["count"])), reverse=True)
        return ranked[: self.MAX_COMMUNITIES]

    def _compare_communities(self, context: dict[str, Any], query_profile: dict[str, Any]) -> dict[str, Any] | None:
        communities = context["communities"][:6]
        if len(communities) < 2:
            return None

        now = datetime.now(UTC)
        scored_items = []
        for community in communities:
            cid = self._to_int(community.get("id"))
            posts = [post for post in context["posts"] if self._post_community_id(post) == cid]
            avg_vote = (sum(int(post.get("voteScore", 0)) for post in posts) / len(posts)) if posts else 0.0
            avg_comments = (sum(int(post.get("commentCount", 0)) for post in posts) / len(posts)) if posts else 0.0
            member_count = int(community.get("memberCount", 0) or 0)
            post_density = 0.0
            created_at = self._parse_dt(community.get("createdAt"))
            if created_at is not None:
                age_days = max((now - created_at).days, 1)
                post_density = (len(posts) / age_days) * 30
            score = avg_vote * 1.2 + avg_comments * 1.4 + (member_count / 1000) + (post_density * 2.0)
            scored_items.append(
                {
                    "id": cid,
                    "name": community.get("name"),
                    "memberCount": member_count,
                    "samplePosts": len(posts),
                    "avgVoteScore": round(avg_vote, 2),
                    "avgCommentCount": round(avg_comments, 2),
                    "post_density": round(post_density, 3),
                    "score": round(score, 3),
                }
            )

        scored_items.sort(key=lambda item: item["score"], reverse=True)
        summary = f"{scored_items[0]['name']} leads on overall activity signal against {scored_items[1]['name']}."
        context["ranking_factors"].append(
            {
                "factor": "community_comparison",
                "weights": {"votes": 1.2, "comments": 1.4, "members_k": 1.0, "post_density": 2.0},
            }
        )
        return {"summary": summary, "items": scored_items[:4], "mode": "side_by_side"}

    def _rank_results(
        self,
        context: dict[str, Any],
        query_profile: dict[str, Any],
        community_index: dict[int, dict[str, Any]],
    ) -> dict[str, int]:
        post_scores: list[tuple[float, dict[str, Any], str]] = []
        freshness_days = query_profile.get("freshness_days")
        now = datetime.now(UTC)
        matched_comment_post_ids = {
            post_id
            for post_id in (self._to_int(comment.get("postId")) for comment in context["matched_comments"])
            if post_id is not None
        }

        for post in context["posts"]:
            votes = int(post.get("voteScore", 0))
            comments = int(post.get("commentCount", 0))
            views = int(post.get("viewCount", 0))
            freshness_bonus = 0.0
            created = self._parse_dt(post.get("createdAt"))
            if freshness_days and created:
                age_days = max(0.0, (now - created).total_seconds() / 86400)
                freshness_bonus = max(0.0, 10.0 - age_days)
            matched_comment_bonus = 15.0 if (self._to_int(post.get("id")) or 0) in matched_comment_post_ids else 0.0
            score = votes * 1.5 + comments * 2.0 + min(views / 100, 10) + freshness_bonus + matched_comment_bonus
            post_id = self._to_int(post.get("id")) or 0
            why = (
                f"votes={votes}, comments={comments}, freshness_bonus={round(freshness_bonus, 2)}, "
                f"matched_comment_bonus={round(matched_comment_bonus, 2)}"
            )
            post_scores.append((score, post, why))
            context["post_scores"][post_id] = round(score, 3)
            context["why_post"][post_id] = why

        post_scores.sort(key=lambda item: item[0], reverse=True)
        context["posts"] = [item[1] for item in post_scores[: self.MAX_POSTS]]

        community_scores: list[tuple[float, dict[str, Any], str]] = []
        for community in context["communities"]:
            cid = self._to_int(community.get("id"))
            related_posts = [post for post in context["posts"] if self._post_community_id(post) == cid]
            member_count = int(community.get("memberCount", 0) or 0)
            post_signal = sum(context["post_scores"].get(self._to_int(post.get("id")) or 0, 0.0) for post in related_posts[:6])
            post_density = 0.0
            created_at = self._parse_dt(community.get("createdAt"))
            if created_at is not None:
                age_days = max((now - created_at).days, 1)
                post_density = (len(related_posts) / age_days) * 30
            score = (member_count / 1500) + post_signal + (post_density * 2.0)
            why = f"member_count={member_count}, post_signal={round(post_signal, 2)}, post_density={round(post_density, 3)}"
            community_scores.append((score, community, why))
            if cid is not None:
                context["community_scores"][cid] = round(score, 3)
                context["why_community"][cid] = why

        community_scores.sort(key=lambda item: item[0], reverse=True)
        context["communities"] = [item[1] for item in community_scores[: self.MAX_COMMUNITIES]]
        context["ranking_factors"].append(
            {
                "factor": "balanced_ranking",
                "post_weights": {
                    "voteScore": 1.5,
                    "commentCount": 2.0,
                    "viewCount": 0.01,
                    "matched_comment_bonus": 15.0,
                },
                "community_weights": {"memberCount": 0.00066, "postSignal": 1.0, "post_density": 2.0},
                "freshness_days": freshness_days,
            }
        )
        return {"posts_ranked": len(post_scores), "communities_ranked": len(community_scores)}

    def _build_cited_summary(self, context: dict[str, Any]) -> str:
        posts = context["posts"][:3]
        comments = context["matched_comments"][:2]
        citations = []
        for post in posts:
            post_id = self._to_int(post.get("id"))
            title = str(post.get("title", "")).strip()[:90]
            citations.append(f"post#{post_id}: {title}")
        for comment in comments:
            comment_id = self._to_int(comment.get("id"))
            post_id = self._to_int(comment.get("postId"))
            citations.append(f"comment#{comment_id}@post#{post_id}")
        if not citations:
            return ""
        return "Evidence summary: " + "; ".join(citations[:6])

    def _extract_actionable_advice(self, context: dict[str, Any]) -> list[str]:
        advice: list[str] = []
        top_flairs = self._build_flair_trends(context["posts"])[:3]
        for trend in top_flairs:
            advice.append(f"Prioritize content under flair '{trend['name']}' (recent={trend['recent']}, total={trend['count']}).")
        if context["community_comparisons"]:
            top = context["community_comparisons"][0].get("items", [])
            if top:
                advice.append(f"Start in {top[0].get('name')} for higher engagement signal.")
        if context["matched_comments"]:
            advice.append("Review top matched comments to capture nuanced constraints before posting.")
        return advice[:5]

    def _collect_user_context(self, posts: list[dict[str, Any]], comments_by_post: dict[str, Any]) -> list[str]:
        users: set[str] = set()
        for post in posts:
            author = str(post.get("authorName", "")).strip()
            if author:
                users.add(author)
        for key, comment_list in comments_by_post.items():
            post_id = self._to_int(key) or 0
            for comment in self._flatten_comments(comment_list, post_id=post_id):
                author = str(comment.get("authorName", "")).strip()
                if author:
                    users.add(author)
        return sorted(users)[: self.MAX_POSTS]

    def _infer_result_type(self, lowered_query: str) -> str:
        if any(word in lowered_query for word in ["flairs", "tags", "labels"]):
            return "flairs"
        if any(word in lowered_query for word in ["rules", "policies", "guidelines"]):
            return "rules"
        if any(word in lowered_query for word in ["communities", "groups", "communities with", "communities about"]):
            return "communities"
        if any(word in lowered_query for word in ["posts", "post from", "posts about", "posts by"]):
            return "posts"
        return "all"

    def _extract_freshness_days(self, lowered_query: str) -> int | None:
        direct = re.search(r"(?:last|past)\s+(\d{1,3})\s+(day|days|week|weeks|month|months)", lowered_query)
        if direct:
            value = int(direct.group(1))
            unit = direct.group(2)
            if unit.startswith("week"):
                return value * 7
            if unit.startswith("month"):
                return value * 30
            return value
        if "last week" in lowered_query or "past week" in lowered_query:
            return 7
        if "last month" in lowered_query or "past month" in lowered_query:
            return 30
        if "today" in lowered_query:
            return 1
        return None

    def _coerce_days(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            parsed = int(value)
            return parsed if parsed > 0 else None
        except (TypeError, ValueError):
            return None

    def _coerce_confidence(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(parsed, 1.0))

    def _parse_dt(self, value: Any) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None

    def _filter_posts_by_freshness(self, posts: list[dict[str, Any]], freshness_days: int | None) -> list[dict[str, Any]]:
        if freshness_days is None:
            return posts
        cutoff = datetime.now(UTC) - timedelta(days=freshness_days)
        filtered = []
        for post in posts:
            created = self._parse_dt(post.get("createdAt"))
            if created is None or created >= cutoff:
                filtered.append(post)
        return filtered

    def _refresh_matched_comments_from_query(self, context: dict[str, Any], query: str) -> None:
        tokens = set(self._tokenize(query))
        if not tokens:
            return
        matched = list(context["matched_comments"])
        for key, comments in context["comments_by_post"].items():
            post_id = self._to_int(key) or 0
            for comment in self._flatten_comments(comments, post_id=post_id):
                content = str(comment.get("content", "")).lower()
                overlap = sum(1 for token in tokens if token in content)
                if overlap > 0:
                    comment["score"] = overlap
                    matched.append(comment)
        self._merge_comments(context["matched_comments"], matched)

    def _normalize_follow_ups(self, raw: Any, normalized_query: str) -> list[str]:
        follow_ups: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                text = str(item).strip()
                if text:
                    follow_ups.append(text)
                if len(follow_ups) == 3:
                    break
        if not follow_ups:
            follow_ups = [
                f"Show me top posts about {normalized_query}",
                f"Compare two communities focused on {normalized_query}",
                f"What are the most actionable next steps for {normalized_query}?",
            ]
        return follow_ups

    def _clamp_confidence(self, value: Any, evidence_count: int) -> float:
        try:
            parsed = float(value)
            if 0.0 <= parsed <= 1.0:
                return parsed
        except (TypeError, ValueError):
            pass
        if evidence_count >= 30:
            return 0.88
        if evidence_count >= 15:
            return 0.74
        if evidence_count >= 6:
            return 0.6
        if evidence_count >= 1:
            return 0.45
        return 0.25

    def _normalize_gaps(self, raw: Any, context: dict[str, Any], query_profile: dict[str, Any]) -> list[str]:
        gaps: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                text = str(item).strip()
                if text:
                    gaps.append(text)
                if len(gaps) == 5:
                    break
        if not context["matched_comments"]:
            gaps.append("Limited direct comment evidence for this query.")
        if query_profile.get("freshness_days") and not self._filter_posts_by_freshness(context["posts"], query_profile["freshness_days"]):
            gaps.append("Very few recent posts matched the requested freshness window.")
        if not context["communities"]:
            gaps.append("No strong community-level matches were found.")
        deduped: list[str] = []
        seen: set[str] = set()
        for gap in gaps:
            key = gap.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(gap)
        return deduped[:5]

    def _next_best_actions(self, plan: PlannerResponse, action_budget: int, query_profile: dict[str, Any]) -> list[str]:
        pending = [action.type for action in plan.actions[action_budget:]]
        suggestions: list[str] = []
        if pending:
            suggestions.extend(pending[:4])
        if query_profile["comparison_mode"] and "compare_communities" not in suggestions:
            suggestions.append("compare_communities")
        if query_profile["recommendation_mode"] and "rank_results" not in suggestions:
            suggestions.append("rank_results")
        if "search_comments" not in suggestions:
            suggestions.append("search_comments")
        return suggestions[:5]
