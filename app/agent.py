from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
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

    def __init__(self, backend: BackendClient, groq_api_key: str, groq_model: str, ollama_base_url: str, ollama_model: str, llm_provider: str, max_actions: int) -> None:
        self.backend = backend
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.ollama_model = ollama_model
        self.llm_provider = llm_provider.lower()
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

        seed_posts = await self.backend.search_posts(query, user_id=user_id, limit=20 if query_profile["recommendation_mode"] else 12)
        if query_profile["author_target"]:
            # Check for month/year in query_profile (simple heuristic for now)
            month, year = None, None
            import re
            m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?", query.lower())
            if m:
                month_str = m.group(1)
                year = int(m.group(2)) if m.group(2) else datetime.now().year
                months = ["january","february","march","april","may","june","july","august","september","october","november","december"]
                month = months.index(month_str) + 1
            author_seed = await self.backend.get_user_posts(str(query_profile["author_target"]), user_id=user_id, limit=16, month=month, year=year)
            self._merge_posts(seed_posts, author_seed)
        communities = await self.backend.list_communities(user_id=user_id)

        if is_scoped_search:
            trace.append({"step": "scope_info", "community_id": community_id, "scope": "single_community"})
            seed_posts = [p for p in seed_posts if self._post_community_id(p) == community_id]
            communities = [c for c in communities if self._to_int(c.get("id")) == community_id]

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

        plan = await self._plan_actions(query=query, seed_posts=seed_posts, communities=communities, action_budget=action_budget)
        plan = self._augment_plan_for_profile(plan=plan, query=query, query_profile=query_profile, action_budget=action_budget)
        plan = self._apply_mode_enhancements(plan=plan, query=query, query_profile=query_profile, action_budget=action_budget)

        if is_scoped_search:
            filtered_actions = [a for a in plan.actions if a.type != "compare_communities"]
            trace.append(
                {
                    "step": "plan_optimization",
                    "reason": "scoped_search_context",
                    "actions_before": len(plan.actions),
                    "actions_after": len(filtered_actions),
                }
            )
            plan.actions = filtered_actions

        trace.append({"step": "plan", "normalized_query": plan.normalized_query, "actions": [a.model_dump() for a in plan.actions]})
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

        self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
        if community_focus:
            context["posts"] = []

        for action in plan.actions[:action_budget]:
            result = await self._execute_action(
                action=action,
                context=context,
                user_id=user_id,
                community_index=community_index,
                community_focus=community_focus,
                query_profile=query_profile,
            )
            trace.append({"step": "action", "action": action.model_dump(), "result": result})

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
        self, query: str, seed_posts: list[dict[str, Any]], communities: list[dict[str, Any]], action_budget: int
    ) -> PlannerResponse:
        prompt = (
            "You are an execution planner for community search.\n"
            "Return ONLY JSON with schema:\n"
            "{"
            '"normalized_query":"...",'
            '"intent":"...",'
            '"actions":[{"type":"search_posts|search_comments|get_user_posts|get_user_comments|get_post_by_id|get_related_posts|get_flair_trends|compare_communities|rank_results|summarize_with_citations|extract_actionable_advice|search_flairs|search_rules|list_communities|get_community_flairs|get_post_comments|get_trending_posts","args":{},"reason":"..."}]'
            "}\n"
            f"Max actions: {action_budget}.\n"
            "Prioritize multi-hop actions when user asks for analysis/comparison/recommendation/explanation.\n"
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
                AgentAction(type="list_communities", args={"query": query, "limit": 14}, reason="Find best matching communities"),
                AgentAction(type="search_comments", args={"query": query, "limit_posts": 8, "limit_comments_per_post": 24}, reason="Find direct discussion evidence"),
                AgentAction(type="search_flairs", args={"query": query}, reason="Find relevant post flairs/tags"),
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
        try:
            if action.type == "search_posts":
                q = str(action.args.get("query", "")).strip()
                if not q:
                    return {"ok": False, "error": "missing query"}
                limit = int(action.args.get("limit", 16))
                posts = await self.backend.search_posts(q, user_id=user_id, limit=limit)
                posts = self._filter_posts_by_freshness(posts, query_profile["freshness_days"])
                self._merge_posts(context["posts"], posts)
                self._collect_flairs(context["flairs"], posts)
                self._apply_query_constraints(context=context, query_profile=query_profile, community_index=community_index)
                return {"ok": True, "posts_added": len(posts)}

            if action.type == "search_comments":
                q = str(action.args.get("query", "")).strip()
                if not q:
                    return {"ok": False, "error": "missing query"}
                limit_posts = int(action.args.get("limit_posts", 6))
                limit_comments = int(action.args.get("limit_comments_per_post", 20))
                comments = await self._search_comments(query=q, posts=context["posts"][: max(1, min(limit_posts, 12))], user_id=user_id, limit_per_post=limit_comments)
                self._merge_comments(context["matched_comments"], comments)
                return {"ok": True, "comments_found": len(comments)}

            if action.type == "get_user_posts":
                username = str(action.args.get("username", query_profile.get("author_target", ""))).strip()
                if not username:
                    return {"ok": False, "error": "missing username"}
                limit = int(action.args.get("limit", 20))
                posts = await self.backend.get_user_posts(username, user_id=user_id, limit=limit)
                posts = self._filter_posts_by_freshness(posts, query_profile["freshness_days"])
                self._merge_posts(context["posts"], posts)
                self._collect_flairs(context["flairs"], posts)
                return {"ok": True, "posts_added": len(posts), "username": username}

            if action.type == "get_user_comments":
                username = str(action.args.get("username", query_profile.get("author_target", ""))).strip().lower()
                if not username:
                    return {"ok": False, "error": "missing username"}
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

            if action.type == "get_post_by_id":
                post_id = int(action.args.get("post_id", 0))
                if post_id <= 0:
                    return {"ok": False, "error": "missing post_id"}
                post = await self.backend.get_post_by_id(post_id, user_id=user_id)
                if not post:
                    return {"ok": True, "post_found": False}
                self._merge_posts(context["posts"], [post])
                self._collect_flairs(context["flairs"], [post])
                return {"ok": True, "post_found": True}

            if action.type == "get_related_posts":
                post_id = int(action.args.get("post_id", 0))
                if post_id <= 0:
                    return {"ok": False, "error": "missing post_id"}
                related = await self._get_related_posts(post_id=post_id, context=context, user_id=user_id, limit=int(action.args.get("limit", 8)))
                self._merge_posts(context["posts"], related)
                self._collect_flairs(context["flairs"], related)
                return {"ok": True, "related_added": len(related)}

            if action.type == "get_flair_trends":
                trends = self._build_flair_trends(context["posts"])
                for trend in trends:
                    context["flairs"].append(trend["name"])
                context["ranking_factors"].append({"factor": "flair_trends", "top": trends[:5]})
                return {"ok": True, "trends_found": len(trends)}

            if action.type == "compare_communities":
                comparison = self._compare_communities(context=context, query_profile=query_profile)
                if comparison:
                    context["community_comparisons"].append(comparison)
                    context["analysis_notes"].append(f"Comparison: {comparison.get('summary', '')}".strip())
                    return {"ok": True, "compared": len(comparison.get("items", []))}
                return {"ok": True, "compared": 0}

            if action.type == "rank_results":
                ranked = self._rank_results(context=context, query_profile=query_profile, community_index=community_index)
                return {"ok": True, **ranked}

            if action.type == "summarize_with_citations":
                note = self._build_cited_summary(context=context)
                if note:
                    context["analysis_notes"].append(note)
                    return {"ok": True, "summary_added": True}
                return {"ok": True, "summary_added": False}

            if action.type == "extract_actionable_advice":
                advice = self._extract_actionable_advice(context=context)
                if advice:
                    context["actionable_advice"].extend(advice)
                return {"ok": True, "advice_items": len(advice)}

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
                else:
                    context["communities"] = communities[: max(1, min(limit, 30))]
                context["selected_community_ids"] = {
                    cid for cid in (self._to_int(c.get("id")) for c in context["communities"]) if cid is not None
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
                self._refresh_matched_comments_from_query(context=context, query=str(action.args.get("query", "")))
                return {"ok": True, "comments_loaded": len(comments)}

            if action.type == "get_trending_posts":
                sort = str(action.args.get("sort", "HOT")).upper()
                window = str(action.args.get("window", "ALL")).upper()
                limit = int(action.args.get("limit", 8))
                posts = await self.backend.get_trending_posts(user_id=user_id, sort=sort, window=window, limit=limit)
                posts = self._filter_posts_by_freshness(posts, query_profile["freshness_days"])
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
            '{"answer":"...", "follow_ups":["...","...","..."], "confidence":0.0, "gaps":["..."]}\n'
            "Rules:\n"
            "- answer: 3-6 sentences, practical, contextual.\n"
            "- Include comparative/recommendation framing when user intent asks for it.\n"
            "- Mention what drove ranking decisions when recommendation_mode=true.\n"
            "- If evidence is weak, list concrete gaps.\n"
            "- Never invent posts, comments, or metrics.\n\n"
            f"User query: {query}\n"
            f"Normalized query: {normalized_query}\n\n"
            f"Query profile: {json.dumps(query_profile, ensure_ascii=False)}\n"
            f"Posts: {json.dumps(self._trim_posts(context['posts'], context['why_post']), ensure_ascii=False)}\n"
            f"Communities: {json.dumps(self._trim_communities(context['communities'], context['why_community']), ensure_ascii=False)}\n"
            f"Comments: {json.dumps(self._trim_comments(context['matched_comments']), ensure_ascii=False)}\n"
            f"Flairs: {json.dumps(self._trim_flairs(context['flairs']), ensure_ascii=False)}\n"
            f"Rules: {json.dumps(self._trim_rules(context['rules']), ensure_ascii=False)}\n"
            f"Ranking factors: {json.dumps(context['ranking_factors'][:6], ensure_ascii=False)}\n"
            f"Analysis notes: {json.dumps(context['analysis_notes'][:4], ensure_ascii=False)}\n"
            f"Actionable advice: {json.dumps(context['actionable_advice'][:5], ensure_ascii=False)}\n"
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

    async def _groq_json(self, prompt: str) -> str:
        # Try Groq first if selected, fallback to Ollama if fails or if provider is ollama
        if self.llm_provider == "ollama":
            return await self._ollama_json(prompt)
        payload = {
            "model": self.groq_model,
            "temperature": 0.2,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(25)) as client:
                response = await client.post(self.groq_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        except Exception as ex:
            # Fallback to Ollama if Groq fails
            return await self._ollama_json(prompt)

    async def _ollama_json(self, prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "format": "json",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.2, "num_predict": 900}
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(25)) as client:
            response = await client.post(f"{self.ollama_base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            # Ollama returns the result in 'message' or 'response' depending on version
            content = data.get("message", {}).get("content") or data.get("response")
            return str(content or "").strip()

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
        seen = {self._to_int(p.get("id")) for p in base}
        for post in incoming:
            post_id = self._to_int(post.get("id"))
            if post_id in seen:
                continue
            base.append(post)
            seen.add(post_id)
        base.sort(key=lambda p: (int(p.get("voteScore", 0)), int(p.get("commentCount", 0))), reverse=True)
        del base[40:]

    def _merge_comments(self, base: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> None:
        seen = {(self._to_int(c.get("postId")), self._to_int(c.get("id"))) for c in base}
        for comment in incoming:
            key = (self._to_int(comment.get("postId")), self._to_int(comment.get("id")))
            if key in seen:
                continue
            base.append(comment)
            seen.add(key)
        base.sort(key=lambda c: int(c.get("score", 0)), reverse=True)
        del base[60:]

    def _collect_flairs(self, flair_list: list[str], posts: list[dict[str, Any]]) -> None:
        for post in posts:
            flair = str(post.get("flairName", "")).strip()
            if flair:
                flair_list.append(flair)

    def _trim_posts(self, posts: list[dict[str, Any]], why_map: dict[int, str] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for post in posts[:20]:
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
        for community in communities[:20]:
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
        for comment in comments[:40]:
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
            if len(out) == 20:
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

    async def _auto_load_flairs(self, context: dict[str, Any], user_id: int | None, trace: list[dict[str, Any]]) -> None:
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

    async def _ensure_comment_context(self, context: dict[str, Any], user_id: int | None, trace: list[dict[str, Any]]) -> None:
        top_posts = sorted(
            context["posts"],
            key=lambda post: (int(post.get("voteScore", 0)), int(post.get("commentCount", 0))),
            reverse=True,
        )[:6]
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

    async def _build_query_profile(self, query: str) -> dict[str, Any]:
        lowered = (query or "").strip().lower()
        llm_profile = await self._interpret_query_with_llm(query)
        if llm_profile:
            return llm_profile

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
        }

    async def _interpret_query_with_llm(self, query: str) -> dict[str, Any] | None:
        prompt = (
            "Analyze this community search query and extract key information.\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{"
            '"author_target": null or "username",'
            '"animals": [] or ["dog", "cat"] etc.,'
            '"search_query": "what to actually search for",'
            '"is_author_query": true/false,'
            '"is_animal_query": true/false,'
            '"result_type": "posts" or "communities" or "flairs" or "rules" or "all",'
            '"freshness_days": null or integer,'
            '"comparison_mode": true/false,'
            '"recommendation_mode": true/false,'
            '"explainability_mode": true/false'
            "}\n\n"
            "Rules:\n"
            "- author_target: extract username for 'posts by/from someone'.\n"
            "- animals: dog, cat, bird, fish, rabbit, hamster, guinea_pig, parrot.\n"
            "- freshness_days: parse phrases like 'last 7 days', 'past month' into days.\n"
            "- comparison_mode: true when user asks to compare options.\n"
            "- recommendation_mode: true when user asks for best/top/ranked picks.\n"
            "- explainability_mode: true when user asks why/reasons OR recommendations are requested.\n"
            "- result_type defaults to 'all' when unclear.\n\n"
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
            if freshness_days is None:
                freshness_days = self._extract_freshness_days((query or "").lower())

            recommendation_mode = bool(parsed.get("recommendation_mode", False))
            explainability_mode = bool(parsed.get("explainability_mode", False)) or recommendation_mode
            comparison_mode = bool(parsed.get("comparison_mode", False))
            is_author = bool(parsed.get("is_author_query", False))
            is_animal = bool(parsed.get("is_animal_query", False))

            return {
                "author_target": author_target,
                "strict_author": is_author and author_target is not None,
                "animals": animals,
                "strict_animals": is_animal and bool(animals),
                "topic_tokens": self._tokenize(search_query or "") if search_query else [],
                "result_type": result_type,
                "freshness_days": freshness_days,
                "comparison_mode": comparison_mode,
                "recommendation_mode": recommendation_mode,
                "explainability_mode": explainability_mode,
                "multi_hop_mode": True,
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
            if not any(a.type == "get_user_posts" for a in actions):
                actions.insert(0, AgentAction(type="get_user_posts", args={"username": author_q, "limit": 18}, reason="Collect author-specific posts"))
            if not any(a.type == "get_user_comments" for a in actions):
                actions.append(AgentAction(type="get_user_comments", args={"username": author_q}, reason="Collect author-specific comment evidence"))

        if query_profile["animals"]:
            animal_q = str(query_profile["animals"][0])
            if not any(a.type == "search_posts" and animal_q in str(a.args.get("query", "")).lower() for a in actions):
                actions.append(AgentAction(type="search_posts", args={"query": animal_q, "limit": 14}, reason="Fetch species-focused context"))

        if query_profile["freshness_days"] is not None and not any(a.type == "rank_results" for a in actions):
            actions.append(AgentAction(type="rank_results", args={"strategy": "freshness_weighted"}, reason="Enforce freshness-aware ranking"))

        if not actions:
            actions = self._default_plan(query).actions
        return PlannerResponse(normalized_query=plan.normalized_query, intent=plan.intent, actions=actions[:action_budget])

    def _apply_mode_enhancements(self, plan: PlannerResponse, query: str, query_profile: dict[str, Any], action_budget: int) -> PlannerResponse:
        actions = list(plan.actions)
        if query_profile["multi_hop_mode"]:
            if not any(a.type == "search_comments" for a in actions):
                actions.append(AgentAction(type="search_comments", args={"query": query, "limit_posts": 6}, reason="Add discussion depth for multi-hop reasoning"))
        if query_profile["comparison_mode"] and not any(a.type == "compare_communities" for a in actions):
            actions.append(AgentAction(type="compare_communities", args={}, reason="Generate side-by-side community comparison"))
        if query_profile["recommendation_mode"] and not any(a.type == "rank_results" for a in actions):
            actions.append(AgentAction(type="rank_results", args={"strategy": "balanced"}, reason="Rank best options for recommendation"))
        if not any(a.type == "summarize_with_citations" for a in actions):
            actions.append(AgentAction(type="summarize_with_citations", args={}, reason="Provide citation-grounded summary notes"))
        if not any(a.type == "extract_actionable_advice" for a in actions):
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
        self, context: dict[str, Any], query_profile: dict[str, Any], community_index: dict[int, dict[str, Any]]
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
                context["communities"] = scoped_communities[:20]

        context["posts"] = self._filter_posts_by_freshness(context["posts"], query_profile.get("freshness_days"))
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
        context["posts"] = filtered[:40]

    async def _search_comments(
        self, query: str, posts: list[dict[str, Any]], user_id: int | None, limit_per_post: int = 20
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
                results = sorted(results, key=lambda c: int(c.get("score", 0)), reverse=True)[: max(1, limit_per_post * len(posts))]
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
        title_tokens = [t for t in self._tokenize(str(target.get("title", ""))) if t not in self.STOP_WORDS][:4]
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
            if len(dedup) >= max(1, min(limit, 20)):
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
        ranked = sorted(trends.values(), key=lambda t: (int(t["recent"]), int(t["count"])), reverse=True)
        return ranked[:20]

    def _compare_communities(self, context: dict[str, Any], query_profile: dict[str, Any]) -> dict[str, Any] | None:
        communities = context["communities"][:6]
        if len(communities) < 2:
            return None

        scored_items = []
        for community in communities:
            cid = self._to_int(community.get("id"))
            posts = [p for p in context["posts"] if self._post_community_id(p) == cid]
            avg_vote = (sum(int(p.get("voteScore", 0)) for p in posts) / len(posts)) if posts else 0.0
            avg_comments = (sum(int(p.get("commentCount", 0)) for p in posts) / len(posts)) if posts else 0.0
            member_count = int(community.get("memberCount", 0) or 0)
            score = avg_vote * 1.2 + avg_comments * 1.4 + (member_count / 1000)
            scored_items.append(
                {
                    "id": cid,
                    "name": community.get("name"),
                    "memberCount": member_count,
                    "samplePosts": len(posts),
                    "avgVoteScore": round(avg_vote, 2),
                    "avgCommentCount": round(avg_comments, 2),
                    "score": round(score, 3),
                }
            )

        scored_items.sort(key=lambda item: item["score"], reverse=True)
        summary = f"{scored_items[0]['name']} leads on overall activity signal against {scored_items[1]['name']}."
        context["ranking_factors"].append({"factor": "community_comparison", "weights": {"votes": 1.2, "comments": 1.4, "members_k": 1.0}})
        return {"summary": summary, "items": scored_items[:4], "mode": "side_by_side"}

    def _rank_results(
        self, context: dict[str, Any], query_profile: dict[str, Any], community_index: dict[int, dict[str, Any]]
    ) -> dict[str, int]:
        post_scores: list[tuple[float, dict[str, Any], str]] = []
        freshness_days = query_profile.get("freshness_days")
        now = datetime.now(UTC)
        for post in context["posts"]:
            votes = int(post.get("voteScore", 0))
            comments = int(post.get("commentCount", 0))
            views = int(post.get("viewCount", 0))
            freshness_bonus = 0.0
            created = self._parse_dt(post.get("createdAt"))
            if freshness_days and created:
                age_days = max(0.0, (now - created).total_seconds() / 86400)
                freshness_bonus = max(0.0, 10.0 - age_days)
            score = votes * 1.5 + comments * 2.0 + min(views / 100, 10) + freshness_bonus
            post_id = self._to_int(post.get("id")) or 0
            why = f"votes={votes}, comments={comments}, freshness_bonus={round(freshness_bonus, 2)}"
            post_scores.append((score, post, why))
            context["post_scores"][post_id] = round(score, 3)
            context["why_post"][post_id] = why

        post_scores.sort(key=lambda item: item[0], reverse=True)
        context["posts"] = [item[1] for item in post_scores[:40]]

        community_scores: list[tuple[float, dict[str, Any], str]] = []
        for community in context["communities"]:
            cid = self._to_int(community.get("id"))
            related_posts = [p for p in context["posts"] if self._post_community_id(p) == cid]
            member_count = int(community.get("memberCount", 0) or 0)
            post_signal = sum(context["post_scores"].get(self._to_int(p.get("id")) or 0, 0.0) for p in related_posts[:6])
            score = (member_count / 1500) + post_signal
            why = f"member_count={member_count}, post_signal={round(post_signal, 2)}"
            community_scores.append((score, community, why))
            if cid is not None:
                context["community_scores"][cid] = round(score, 3)
                context["why_community"][cid] = why

        community_scores.sort(key=lambda item: item[0], reverse=True)
        context["communities"] = [item[1] for item in community_scores[:20]]
        context["ranking_factors"].append(
            {
                "factor": "balanced_ranking",
                "post_weights": {"voteScore": 1.5, "commentCount": 2.0, "viewCount": 0.01},
                "community_weights": {"memberCount": 0.00066, "postSignal": 1.0},
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
        for comment_list in comments_by_post.values():
            for comment in self._flatten_comments(comment_list, post_id=0):
                author = str(comment.get("authorName", "")).strip()
                if author:
                    users.add(author)
        return sorted(users)[:40]

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
