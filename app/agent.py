from __future__ import annotations

import json
from typing import Any

import httpx

from .backend_client import BackendClient
from .models import AgentAction, PlannerResponse


class CommunitySearchAgent:
    def __init__(
        self,
        backend: BackendClient,
        groq_api_key: str,
        groq_model: str,
        max_actions: int,
    ) -> None:
        self.backend = backend
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.max_actions = max_actions
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"

    async def run(
        self,
        query: str,
        user_id: int | None = None,
        act_as_user_id: int | None = None,
        max_actions: int | None = None,
    ) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        action_budget = max_actions or self.max_actions

        seed_posts = await self.backend.search_posts(query, user_id=user_id, limit=12)
        communities = await self.backend.list_communities(user_id=user_id)
        community_index = self._community_index(communities)
        trace.append({"step": "seed_fetch", "posts": len(seed_posts), "communities": len(communities)})

        plan = await self._plan_actions(query=query, seed_posts=seed_posts, communities=communities, action_budget=action_budget)
        trace.append({"step": "plan", "normalized_query": plan.normalized_query, "actions": [a.model_dump() for a in plan.actions]})
        community_focus = self._is_community_intent(query=query, intent=plan.intent)

        context = {
            "posts": list(seed_posts),
            "communities": communities,
            "flairs": [],
            "comments_by_post": {},
            "selected_community_ids": set(),
            "flairs_loaded_for": set(),
        }
        if community_focus:
            # For "find communities" intent, avoid polluting synthesis with unrelated seed posts.
            context["posts"] = []

        for action in plan.actions[:action_budget]:
            result = await self._execute_action(
                action,
                context=context,
                user_id=user_id,
                community_index=community_index,
                community_focus=community_focus,
            )
            trace.append({"step": "action", "action": action.model_dump(), "result": result})

        # Auto-load flair context for top selected communities so responses stay community-grounded.
        selected_community_ids = list(context["selected_community_ids"])[:3]
        for community_id in selected_community_ids:
            if community_id in context["flairs_loaded_for"]:
                continue
            try:
                flairs = await self.backend.list_flairs(community_id)
                names = [str(item.get("name", "")).strip() for item in flairs if item.get("name")]
                context["flairs"].extend(names)
                context["flairs_loaded_for"].add(community_id)
                trace.append({"step": "auto_flairs", "community_id": community_id, "flairs_added": len(names)})
            except Exception as ex:
                fallback_names = await self._fallback_flairs_from_posts(community_id=community_id, user_id=user_id)
                context["flairs"].extend(fallback_names)
                if fallback_names:
                    context["flairs_loaded_for"].add(community_id)
                trace.append(
                    {
                        "step": "auto_flairs",
                        "community_id": community_id,
                        "error": str(ex),
                        "fallback_flairs_added": len(fallback_names),
                    }
                )

        if community_focus and selected_community_ids:
            self._filter_posts_by_selected_communities(context["posts"], context["selected_community_ids"])

        await self._ensure_comment_context(context=context, user_id=user_id, trace=trace)

        synthesis = await self._synthesize(
            query=query,
            normalized_query=plan.normalized_query,
            context=context,
        )
        trace.append({"step": "synthesis", "model": synthesis.get("model", self.groq_model)})

        return {
            "query": query,
            "normalized_query": plan.normalized_query,
            "answer": synthesis["answer"],
            "follow_ups": synthesis["follow_ups"],
            "referenced_posts": self._trim_posts(context["posts"]),
            "referenced_communities": self._trim_communities(context["communities"]),
            "referenced_flairs": sorted({f for f in context["flairs"] if f})[:20],
            "model": synthesis.get("model", self.groq_model),
            "trace": trace,
        }

    async def _plan_actions(
        self, query: str, seed_posts: list[dict[str, Any]], communities: list[dict[str, Any]], action_budget: int
    ) -> PlannerResponse:
        prompt = (
            "You are an execution planner for community search.\n"
            "Return ONLY JSON with schema:\n"
            "{"
            '"normalized_query":"...",'
            '"intent":"...",'
            '"actions":[{"type":"search_posts|list_communities|get_community_flairs|get_post_comments|get_trending_posts","args":{},"reason":"..."}]'
            "}\n"
            f"Max actions: {action_budget}.\n"
            "Prefer actions that increase contextual relevance.\n"
            "Do not exceed max actions.\n\n"
            f"User query: {query}\n\n"
            f"Seed posts sample: {json.dumps(seed_posts[:5], ensure_ascii=False)}\n"
            f"Communities sample: {json.dumps(communities[:8], ensure_ascii=False)}"
        )
        raw = await self._groq_json(prompt)
        parsed = self._parse_json_candidate(raw)
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
                AgentAction(type="list_communities", args={"query": query, "limit": 12}, reason="Find best matching communities"),
                AgentAction(type="get_trending_posts", args={"sort": "HOT", "window": "WEEK", "limit": 8}, reason="Add current high-signal context"),
            ],
        )

    async def _execute_action(
        self,
        action: AgentAction,
        context: dict[str, Any],
        user_id: int | None,
        community_index: dict[int, dict[str, Any]],
        community_focus: bool,
    ) -> dict[str, Any]:
        try:
            if action.type == "search_posts":
                q = str(action.args.get("query", "")).strip()
                if not q:
                    return {"ok": False, "error": "missing query"}
                limit = int(action.args.get("limit", 16))
                posts = await self.backend.search_posts(q, user_id=user_id, limit=limit)
                self._merge_posts(context["posts"], posts)
                self._collect_flairs(context["flairs"], posts)
                return {"ok": True, "posts_added": len(posts)}

            if action.type == "list_communities":
                communities = context["communities"]
                q = str(action.args.get("query", "")).strip().lower()
                limit = int(action.args.get("limit", 12))
                if q:
                    ranked = [
                        c
                        for c in communities
                            if q in str(c.get("name", "")).lower() or q in str(c.get("description", "")).lower()
                    ]
                    context["communities"] = ranked[: max(1, min(limit, 30))]
                    context["selected_community_ids"] = {
                        cid
                        for cid in (self._to_int(c.get("id")) for c in context["communities"])
                        if cid is not None
                    }
                    if community_focus:
                        self._filter_posts_by_selected_communities(context["posts"], context["selected_community_ids"])
                    return {"ok": True, "communities_selected": len(context["communities"])}
                context["communities"] = communities[: max(1, min(limit, 30))]
                context["selected_community_ids"] = {
                    cid
                    for cid in (self._to_int(c.get("id")) for c in context["communities"])
                    if cid is not None
                }
                if community_focus:
                    self._filter_posts_by_selected_communities(context["posts"], context["selected_community_ids"])
                return {"ok": True, "communities_selected": len(context["communities"])}

            if action.type == "get_community_flairs":
                community_id = int(action.args.get("community_id", 0))
                if community_id <= 0:
                    return {"ok": False, "error": "missing community_id"}
                flairs = await self.backend.list_flairs(community_id)
                names = [str(item.get("name", "")).strip() for item in flairs if item.get("name")]
                context["flairs"].extend(names)
                context["flairs_loaded_for"].add(community_id)
                if community_id in community_index:
                    community_index[community_id]["_flairs"] = names
                return {"ok": True, "flairs_added": len(names)}

            if action.type == "get_post_comments":
                post_id = int(action.args.get("post_id", 0))
                if post_id <= 0:
                    return {"ok": False, "error": "missing post_id"}
                comments = await self.backend.get_post_comments(post_id, user_id=user_id)
                context["comments_by_post"][str(post_id)] = comments[:30]
                return {"ok": True, "comments_loaded": len(comments)}

            if action.type == "get_trending_posts":
                sort = str(action.args.get("sort", "HOT")).upper()
                window = str(action.args.get("window", "ALL")).upper()
                limit = int(action.args.get("limit", 8))
                posts = await self.backend.get_trending_posts(user_id=user_id, sort=sort, window=window, limit=limit)
                self._merge_posts(context["posts"], posts)
                self._collect_flairs(context["flairs"], posts)
                return {"ok": True, "trending_added": len(posts)}

            return {"ok": False, "error": f"unsupported action: {action.type}"}
        except Exception as ex:
            return {"ok": False, "error": str(ex)}

    async def _synthesize(self, query: str, normalized_query: str, context: dict[str, Any]) -> dict[str, Any]:
        user_context = self._collect_user_context(context["posts"], context["comments_by_post"])
        prompt = (
            "You are a community assistant. Use only the provided evidence.\n"
            "Return ONLY JSON with schema:\n"
            '{"answer":"...", "follow_ups":["...","...","..."]}\n'
            "Rules:\n"
            "- answer: 3-6 sentences, practical, contextual.\n"
            "- Mention communities/flairs when relevant.\n"
            "- If evidence is weak, say what is missing.\n"
            "- Never invent posts, comments, or metrics.\n\n"
            f"User query: {query}\n"
            f"Normalized query: {normalized_query}\n\n"
            f"Posts: {json.dumps(self._trim_posts(context['posts']), ensure_ascii=False)}\n"
            f"Communities: {json.dumps(self._trim_communities(context['communities']), ensure_ascii=False)}\n"
            f"Flairs: {json.dumps(sorted({f for f in context['flairs'] if f})[:20], ensure_ascii=False)}\n"
            f"Users in scope: {json.dumps(user_context, ensure_ascii=False)}\n"
            f"Comments by post: {json.dumps(context['comments_by_post'], ensure_ascii=False)}\n"
        )

        raw = await self._groq_json(prompt)
        parsed = self._parse_json_candidate(raw)
        if not isinstance(parsed, dict):
            return {
                "answer": raw.strip() or "I could not build a reliable answer from the available context.",
                "follow_ups": [
                    f"Show me top posts about {normalized_query}",
                    f"Find communities focused on {normalized_query}",
                    f"What are common pitfalls for {normalized_query}?",
                ],
                "model": self.groq_model,
            }

        answer = str(parsed.get("answer", "")).strip()
        if not answer:
            answer = "I could not build a reliable answer from the available context."
        follow_ups_raw = parsed.get("follow_ups", [])
        follow_ups: list[str] = []
        if isinstance(follow_ups_raw, list):
            for item in follow_ups_raw:
                text = str(item).strip()
                if text:
                    follow_ups.append(text)
                if len(follow_ups) == 3:
                    break
        if not follow_ups:
            follow_ups = [
                f"Show me top posts about {normalized_query}",
                f"Find communities focused on {normalized_query}",
                f"What are common pitfalls for {normalized_query}?",
            ]

        return {"answer": answer, "follow_ups": follow_ups, "model": self.groq_model}

    async def _groq_json(self, prompt: str) -> str:
        payload = {
            "model": self.groq_model,
            "temperature": 0.2,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(25)) as client:
            response = await client.post(self.groq_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()

    def _parse_json_candidate(self, raw: str) -> dict[str, Any] | None:
        source = (raw or "").strip()
        if not source:
            return None
        source = source.replace("```json", "").replace("```", "").strip()
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
        seen = {p.get("id") for p in base}
        for post in incoming:
            post_id = post.get("id")
            if post_id in seen:
                continue
            base.append(post)
            seen.add(post_id)
        base.sort(key=lambda p: (int(p.get("voteScore", 0)), int(p.get("commentCount", 0))), reverse=True)
        del base[40:]

    def _collect_flairs(self, flair_list: list[str], posts: list[dict[str, Any]]) -> None:
        for post in posts:
            flair = str(post.get("flairName", "")).strip()
            if flair:
                flair_list.append(flair)

    def _trim_posts(self, posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for post in posts[:20]:
            out.append(
                {
                    "id": post.get("id"),
                    "communityId": post.get("communityId"),
                    "communitySlug": post.get("communitySlug"),
                    "title": post.get("title"),
                    "content": str(post.get("content", ""))[:360],
                    "flairName": post.get("flairName"),
                    "type": post.get("type"),
                    "voteScore": post.get("voteScore"),
                    "commentCount": post.get("commentCount"),
                }
            )
        return out

    def _trim_communities(self, communities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for community in communities[:20]:
            out.append(
                {
                    "id": community.get("id"),
                    "name": community.get("name"),
                    "slug": community.get("slug"),
                    "description": str(community.get("description", ""))[:220],
                    "memberCount": community.get("memberCount"),
                    "type": community.get("type"),
                }
            )
        return out

    def _community_index(self, communities: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        index: dict[int, dict[str, Any]] = {}
        for community in communities:
            community_id = community.get("id")
            if isinstance(community_id, int):
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
        posts[:] = [post for post in posts if self._to_int(post.get("communityId")) in selected_ids]

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
                limit=20,
            )
        except Exception:
            return []

        flairs: list[str] = []
        for post in posts:
            flair = str(post.get("flairName", "")).strip()
            if flair:
                flairs.append(flair)
        return flairs

    async def _ensure_comment_context(self, context: dict[str, Any], user_id: int | None, trace: list[dict[str, Any]]) -> None:
        top_posts = sorted(
            context["posts"],
            key=lambda post: (int(post.get("voteScore", 0)), int(post.get("commentCount", 0))),
            reverse=True,
        )[:5]
        for post in top_posts:
            post_id = self._to_int(post.get("id"))
            if post_id is None:
                continue
            key = str(post_id)
            if key in context["comments_by_post"]:
                continue
            try:
                comments = await self.backend.get_post_comments(post_id, user_id=user_id)
                context["comments_by_post"][key] = comments[:40]
                trace.append({"step": "auto_comments", "post_id": post_id, "comments_loaded": len(comments)})
            except Exception as ex:
                trace.append({"step": "auto_comments", "post_id": post_id, "error": str(ex)})

    def _collect_user_context(self, posts: list[dict[str, Any]], comments_by_post: dict[str, Any]) -> list[str]:
        users: set[str] = set()
        for post in posts:
            author = str(post.get("authorName", "")).strip()
            if author:
                users.add(author)
        for comment_list in comments_by_post.values():
            self._collect_comment_authors(comment_list, users)
        return sorted(users)[:40]

    def _collect_comment_authors(self, comments: Any, users: set[str]) -> None:
        if not isinstance(comments, list):
            return
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            author = str(comment.get("authorName", "")).strip()
            if author:
                users.add(author)
            self._collect_comment_authors(comment.get("replies", []), users)
