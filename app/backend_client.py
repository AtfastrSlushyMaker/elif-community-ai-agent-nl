from __future__ import annotations

from typing import Any

import httpx


class BackendClient:
    def __init__(self, base_url: str, community_prefix: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.community_prefix = community_prefix.rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def close(self) -> None:
        await self.client.aclose()

    def _headers(self, user_id: int | None = None, act_as_user_id: int | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if user_id is not None:
            headers["X-User-Id"] = str(user_id)
        if act_as_user_id is not None:
            headers["X-Act-As-User-Id"] = str(act_as_user_id)
        return headers

    async def search_posts(self, query: str, user_id: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base_url}{self.community_prefix}/posts/search",
            params={"q": query},
            headers=self._headers(user_id=user_id),
        )
        response.raise_for_status()
        posts = response.json()
        return posts[: max(1, min(limit, 60))]

    async def get_trending_posts(
        self,
        user_id: int | None = None,
        sort: str = "HOT",
        window: str = "ALL",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base_url}{self.community_prefix}/posts/trending",
            params={"sort": sort, "window": window, "limit": max(1, min(limit, 50))},
            headers=self._headers(user_id=user_id),
        )
        response.raise_for_status()
        return response.json()

    async def list_communities(self, user_id: int | None = None) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base_url}{self.community_prefix}/communities",
            headers=self._headers(user_id=user_id),
        )
        response.raise_for_status()
        return response.json()

    async def list_flairs(self, community_id: int) -> list[dict[str, Any]]:
        response = await self.client.get(f"{self.base_url}{self.community_prefix}/communities/{community_id}/flairs")
        response.raise_for_status()
        return response.json()

    async def get_post_comments(self, post_id: int, user_id: int | None = None) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base_url}{self.community_prefix}/posts/{post_id}/comments",
            headers=self._headers(user_id=user_id),
        )
        response.raise_for_status()
        return response.json()

    async def get_post_by_id(self, post_id: int, user_id: int | None = None) -> dict[str, Any] | None:
        try:
            response = await self.client.get(
                f"{self.base_url}{self.community_prefix}/posts/{post_id}",
                headers=self._headers(user_id=user_id),
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code != 404:
                raise
        posts = await self.search_posts(str(post_id), user_id=user_id, limit=25)
        for post in posts:
            try:
                if int(post.get("id", 0)) == post_id:
                    return post
            except (TypeError, ValueError):
                continue
        return None

    async def get_user_posts(self, username: str, user_id: int | None = None, limit: int = 25) -> list[dict[str, Any]]:
        query = (username or "").strip()
        if not query:
            return []
        posts = await self.search_posts(query, user_id=user_id, limit=max(10, min(limit, 60)))
        lowered = query.lower()
        filtered = [p for p in posts if lowered in str(p.get("authorName", "")).strip().lower()]
        return filtered[: max(1, min(limit, 60))]

    async def get_community_posts(
        self,
        community_id: int,
        user_id: int | None = None,
        sort: str = "HOT",
        window: str = "ALL",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base_url}{self.community_prefix}/communities/{community_id}/posts",
            params={"sort": sort, "window": window},
            headers=self._headers(user_id=user_id),
        )
        response.raise_for_status()
        posts = response.json()
        return posts[: max(1, min(limit, 60))]

    async def get_community_rules(self, community_id: int) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base_url}{self.community_prefix}/communities/{community_id}/rules"
        )
        response.raise_for_status()
        return response.json()

    async def search_flairs_by_name(self, query: str, communities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Search for flairs across communities by name."""
        results = []
        query_lower = query.lower()
        
        for community in communities[:10]:  # Limit to first 10 communities to avoid too many requests
            try:
                community_id = community.get("id")
                if not community_id:
                    continue
                flairs = await self.list_flairs(community_id)
                for flair in flairs:
                    flair_name = str(flair.get("name", "")).lower()
                    if query_lower in flair_name:
                        results.append({
                            "type": "flair",
                            "id": flair.get("id"),
                            "name": flair.get("name"),
                            "community_id": community_id,
                            "community_name": community.get("name"),
                            "color": flair.get("color"),
                            "textColor": flair.get("textColor"),
                        })
            except Exception:
                continue
        
        return results

    async def search_rules_by_content(self, query: str, communities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Search for community rules by content."""
        results = []
        query_lower = query.lower()
        
        for community in communities[:10]:  # Limit to first 10 communities
            try:
                community_id = community.get("id")
                if not community_id:
                    continue
                rules = await self.get_community_rules(community_id)
                for rule in rules:
                    rule_title = str(rule.get("title", "")).lower()
                    rule_description = str(rule.get("description", "")).lower()
                    if query_lower in rule_title or query_lower in rule_description:
                        results.append({
                            "type": "rule",
                            "id": rule.get("id"),
                            "title": rule.get("title"),
                            "description": rule.get("description"),
                            "community_id": community_id,
                            "community_name": community.get("name"),
                            "order": rule.get("order"),
                        })
            except Exception:
                continue
        
        return results
