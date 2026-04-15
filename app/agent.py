from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .backend_client import BackendClient
from .models import AgentAction, PlannerResponse


class CommunitySearchAgent:
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
        community_id: int | None = None,
    ) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        action_budget = max_actions or self.max_actions
        is_scoped_search = community_id is not None
        query_profile = await self._build_query_profile(query)

        if is_scoped_search:
            # Scoped search: only search within a specific community
            trace.append({
                "step": "scope_info",
                "community_id": community_id,
                "scope": "single_community"
            })
            seed_posts = await self.backend.search_posts(query, user_id=user_id, limit=20)
            # Filter posts to only those in the target community
            seed_posts = [p for p in seed_posts if p.get("community_id") == community_id]
            communities = await self.backend.list_communities(user_id=user_id)
            # Filter communities to only the target community
            communities = [c for c in communities if c.get("id") == community_id]
        else:
            # Global search: search across all communities
            seed_posts = await self.backend.search_posts(query, user_id=user_id, limit=12)
            if query_profile["author_target"]:
                author_seed = await self.backend.search_posts(str(query_profile["author_target"]), user_id=user_id, limit=12)
                self._merge_posts(seed_posts, author_seed)
            communities = await self.backend.list_communities(user_id=user_id)
        
        community_index = self._community_index(communities)
        trace.append(
            {
                "step": "seed_fetch",
                "posts": len(seed_posts),
                "communities": len(communities),
                "query_profile": query_profile,
            }
        )

        plan = await self._plan_actions(query=query, seed_posts=seed_posts, communities=communities, action_budget=action_budget)
        plan = self._augment_plan_for_profile(plan=plan, query=query, query_profile=query_profile, action_budget=action_budget)
        
        # For scoped searches, remove unnecessary actions
        if is_scoped_search:
            # No need to search/list other communities in scoped search
            plan.actions = [a for a in plan.actions if a.type not in ["list_communities", "search_flairs", "search_rules"]]
            trace.append({"step": "plan_optimization", "reason": "scoped_search_context", "actions_before": len(plan.actions) + 3, "actions_after": len(plan.actions)})
        
        trace.append({"step": "plan", "normalized_query": plan.normalized_query, "actions": [a.model_dump() for a in plan.actions]})
        community_focus = self._is_community_intent(query=query, intent=plan.intent)

        context = {
            "posts": list(seed_posts),
            "communities": communities,
            "flairs": [],
            "rules": [],
            "comments_by_post": {},
            "selected_community_ids": set(),
            "flairs_loaded_for": set(),
        }
        self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
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
                query_profile=query_profile,
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

        self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
        await self._ensure_comment_context(context=context, user_id=user_id, trace=trace)

        synthesis = await self._synthesize(
            query=query,
            normalized_query=plan.normalized_query,
            context=context,
            query_profile=query_profile,
        )
        trace.append({"step": "synthesis", "model": synthesis.get("model", self.groq_model)})

        # Filter results based on user's result_type intent
        result_type = query_profile.get("result_type", "all")
        referenced_posts = self._trim_posts(context["posts"])
        referenced_communities = self._trim_communities(context["communities"])
        referenced_flairs = sorted({f.get("name", f) if isinstance(f, dict) else f for f in context["flairs"] if f})[:20]
        referenced_rules = context["rules"][:10]  # Limit to 10 rules
        
        # Prepare results based on result_type
        if result_type == "posts":
            # User only wants posts
            referenced_communities = []
            referenced_flairs = []
            referenced_rules = []
            trace.append({"step": "result_filtering", "reason": "result_type is posts", "cleared": "communities,flairs,rules"})
        elif result_type == "communities":
            # User only wants communities
            referenced_posts = []
            referenced_flairs = []
            referenced_rules = []
            trace.append({"step": "result_filtering", "reason": "result_type is communities", "cleared": "posts,flairs,rules"})
        elif result_type == "flairs":
            # User only wants flairs
            referenced_posts = []
            referenced_communities = []
            referenced_rules = []
            trace.append({"step": "result_filtering", "reason": "result_type is flairs", "cleared": "posts,communities,rules"})
        elif result_type == "rules":
            # User only wants rules
            referenced_posts = []
            referenced_communities = []
            referenced_flairs = []
            trace.append({"step": "result_filtering", "reason": "result_type is rules", "cleared": "posts,communities,flairs"})
        # else: result_type == "all" - return everything

        return {
            "query": query,
            "normalized_query": plan.normalized_query,
            "answer": synthesis["answer"],
            "follow_ups": synthesis["follow_ups"],
            "referenced_posts": referenced_posts,
            "referenced_communities": referenced_communities,
            "referenced_flairs": referenced_flairs,
            "referenced_rules": referenced_rules,
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
            '"actions":[{"type":"search_posts|search_flairs|search_rules|list_communities|get_community_flairs|get_post_comments|get_trending_posts","args":{},"reason":"..."}]'
            "}\n"
            f"Max actions: {action_budget}.\n"
            "Available actions:\n"
            "- search_posts: Search for posts matching query\n"
            "- search_flairs: Search for post flairs/tags by name\n"
            "- search_rules: Search for community rules by content\n"
            "- list_communities: Filter communities by query\n"
            "- get_community_flairs: Get all flairs from a specific community\n"
            "- get_post_comments: Get comments from a specific post\n"
            "- get_trending_posts: Get trending posts\n"
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
                AgentAction(type="search_flairs", args={"query": query}, reason="Find relevant post flairs/tags"),
                AgentAction(type="search_rules", args={"query": query}, reason="Find relevant community rules"),
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
        query_profile: dict[str, Any],
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
                self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
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
                    self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
                    return {"ok": True, "communities_selected": len(context["communities"])}
                context["communities"] = communities[: max(1, min(limit, 30))]
                context["selected_community_ids"] = {
                    cid
                    for cid in (self._to_int(c.get("id")) for c in context["communities"])
                    if cid is not None
                }
                if community_focus:
                    self._filter_posts_by_selected_communities(context["posts"], context["selected_community_ids"])
                self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
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
                self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
                return {"ok": True, "trending_added": len(posts)}

            if action.type == "search_flairs":
                q = str(action.args.get("query", "")).strip()
                if not q:
                    return {"ok": False, "error": "missing query"}
                flairs = await self.backend.search_flairs_by_name(q, context["communities"])
                if flairs:
                    context["flairs"].extend([f.get("name") for f in flairs if f.get("name")])
                return {"ok": True, "flairs_found": len(flairs)}

            if action.type == "search_rules":
                q = str(action.args.get("query", "")).strip()
                if not q:
                    return {"ok": False, "error": "missing query"}
                rules = await self.backend.search_rules_by_content(q, context["communities"])
                if rules:
                    context["rules"].extend(rules)
                return {"ok": True, "rules_found": len(rules)}

            return {"ok": False, "error": f"unsupported action: {action.type}"}
        except Exception as ex:
            return {"ok": False, "error": str(ex)}

    async def _synthesize(
        self,
        query: str,
        normalized_query: str,
        context: dict[str, Any],
        query_profile: dict[str, Any],
    ) -> dict[str, Any]:
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
            f"Query profile: {json.dumps(query_profile, ensure_ascii=False)}\n"
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
                    "bannerUrl": community.get("bannerUrl"),
                    "iconUrl": community.get("iconUrl"),
                    "createdAt": community.get("createdAt"),
                    "userRole": community.get("userRole"),
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

    async def _build_query_profile(self, query: str) -> dict[str, Any]:
        lowered = (query or "").strip().lower()
        
        # Try LLM-based interpretation first for better natural language understanding
        llm_profile = await self._interpret_query_with_llm(query)
        if llm_profile:
            return llm_profile
        
        # Fallback to regex-based extraction if LLM interpretation fails
        author_target = self._extract_author_target(lowered)
        animals = sorted(self._extract_animals(lowered))
        topic_tokens = [token for token in self._tokenize(lowered) if token not in self.STOP_WORDS]
        
        # Infer result_type from keywords
        result_type = "all"
        if any(word in lowered for word in ["flairs", "tags", "labels"]):
            result_type = "flairs"
        elif any(word in lowered for word in ["rules", "policies", "guidelines"]):
            result_type = "rules"
        elif any(word in lowered for word in ["communities", "groups", "communities with", "communities about"]):
            result_type = "communities"
        elif any(word in lowered for word in ["posts", "post from", "posts about", "posts by"]):
            result_type = "posts"
        
        return {
            "author_target": author_target,
            "strict_author": author_target is not None,
            "animals": animals,
            "strict_animals": bool(animals),
            "topic_tokens": topic_tokens[:8],
            "result_type": result_type,
        }
    
    async def _interpret_query_with_llm(self, query: str) -> dict[str, Any] | None:
        """Use LLM to interpret user query intent and extract search parameters."""
        prompt = (
            "Analyze this community search query and extract key information.\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{"
            '"author_target": null or "username",'
            '"animals": [] or ["dog", "cat"] etc.,'
            '"search_query": "what to actually search for",'
            '"is_author_query": true/false,'
            '"is_animal_query": true/false,'
            '"result_type": "posts" or "communities" or "flairs" or "rules" or "all"'
            "}\n\n"
            "Rules:\n"
            "- author_target: Extract username if user asks for posts FROM or BY someone. Otherwise null.\n"
            "- animals: Extract pet types mentioned (dog, cat, bird, fish, rabbit, hamster, guinea_pig, parrot).\n"
            "- search_query: The core search keywords, cleaned of stopwords.\n"
            "- is_author_query: true if main intent is finding posts from a specific author.\n"
            "- is_animal_query: true if main intent is about a specific pet type.\n"
            "- result_type: Determine what user wants based on query keywords:\n"
            "  * 'posts' if user says: 'posts about', 'show me posts', 'find posts', 'post from'.\n"
            "  * 'communities' if user says: 'communities about', 'find communities', 'communities with', 'groups about'.\n"
            "  * 'flairs' if user says: 'flairs', 'tags', 'labels'.\n"
            "  * 'rules' if user says: 'rules', 'policies', 'guidelines'.\n"
            "  * 'all' if query is ambiguous or doesn't specify - search across all entities.\n"
            "- Always use lowercase for author_target and animals.\n\n"
            f"User query: {query}"
        )
        try:
            raw = await self._groq_json(prompt)
            parsed = self._parse_json_candidate(raw)
            if not isinstance(parsed, dict):
                return None
            
            author_target = parsed.get("author_target")
            if author_target:
                author_target = str(author_target).strip().lower() or None
            
            animals_raw = parsed.get("animals", [])
            animals = []
            if isinstance(animals_raw, list):
                for animal in animals_raw:
                    animal_str = str(animal).strip().lower()
                    if animal_str:
                        animals.append(animal_str)
            animals = sorted(animals)
            
            search_query = str(parsed.get("search_query", "")).strip() or None
            is_author = bool(parsed.get("is_author_query", False))
            is_animal = bool(parsed.get("is_animal_query", False))
            
            # Extract result_type with validation
            result_type = str(parsed.get("result_type", "all")).strip().lower()
            if result_type not in ("posts", "communities", "flairs", "rules", "all"):
                result_type = "all"
            
            # Fallback search query
            if not search_query:
                search_query = str(author_target or "").strip() if author_target else None
            
            return {
                "author_target": author_target,
                "strict_author": is_author and author_target is not None,
                "animals": animals,
                "strict_animals": is_animal and bool(animals),
                "topic_tokens": self._tokenize(search_query or "") if search_query else [],
                "result_type": result_type,
                "llm_interpreted": True,
            }
        except Exception:
            return None

    def _augment_plan_for_profile(
        self, plan: PlannerResponse, query: str, query_profile: dict[str, Any], action_budget: int
    ) -> PlannerResponse:
        actions = list(plan.actions)
        if query_profile["author_target"]:
            author_q = str(query_profile["author_target"])
            has_author_search = any(
                a.type == "search_posts" and author_q in str(a.args.get("query", "")).lower() for a in actions
            )
            if not has_author_search:
                actions.insert(
                    0,
                    AgentAction(type="search_posts", args={"query": author_q, "limit": 20}, reason="Fetch posts for author intent"),
                )
        if query_profile["animals"]:
            animal_q = str(query_profile["animals"][0])
            has_animal_search = any(
                a.type == "search_posts" and animal_q in str(a.args.get("query", "")).lower() for a in actions
            )
            if not has_animal_search:
                actions.append(
                    AgentAction(
                        type="search_posts",
                        args={"query": animal_q, "limit": 14},
                        reason="Fetch species-focused context",
                    )
                )
            has_community_filter = any(
                a.type == "list_communities" and animal_q in str(a.args.get("query", "")).lower() for a in actions
            )
            if not has_community_filter:
                actions.append(
                    AgentAction(
                        type="list_communities",
                        args={"query": animal_q, "limit": 15},
                        reason="Scope communities by species",
                    )
                )
        if not actions:
            actions = self._default_plan(query).actions
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
        pattern = rf"\b{normalized_alias}\b"
        return re.search(pattern, text.lower()) is not None

    def _post_animals(self, post: dict[str, Any], community_index: dict[int, dict[str, Any]]) -> set[str]:
        community = community_index.get(self._to_int(post.get("communityId")) or -1, {})
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
        self, context: dict[str, Any], query_profile: dict[str, Any], community_index: dict[int, dict[str, Any]]
    ) -> None:
        requested_animals = set(query_profile["animals"])
        if requested_animals and context["communities"]:
            scoped_communities = []
            for community in context["communities"]:
                haystack = " ".join(
                    [
                        str(community.get("name", "")),
                        str(community.get("slug", "")),
                        str(community.get("description", "")),
                    ]
                ).lower()
                if requested_animals.intersection(self._extract_animals(haystack)):
                    scoped_communities.append(community)
            if scoped_communities:
                context["communities"] = scoped_communities[:20]

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
                if author_target in post_author:
                    score += 120
                else:
                    score -= 100

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
            animal_only = [
                post for post in filtered if requested_animals.intersection(self._post_animals(post, community_index))
            ]
            if animal_only:
                filtered = animal_only

        context["posts"] = filtered[:40]
