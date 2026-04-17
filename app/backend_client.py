from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx


class BackendClient:
    def __init__(self, base_url: str, community_prefix: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.community_prefix = community_prefix.rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._cache: dict[str, dict[Any, Any]] = {}
        self.reset_cache()

    def reset_cache(self) -> None:
        self._cache = {
            "list_communities": {},
            "list_flairs": {},
            "get_post_comments": {},
        }

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
        cache_key = user_id
        if cache_key in self._cache["list_communities"]:
            return list(self._cache["list_communities"][cache_key])

        response = await self.client.get(
            f"{self.base_url}{self.community_prefix}/communities",
            headers=self._headers(user_id=user_id),
        )
        response.raise_for_status()
        payload = response.json()
        self._cache["list_communities"][cache_key] = list(payload)
        return list(payload)

    async def list_flairs(self, community_id: int) -> list[dict[str, Any]]:
        if community_id in self._cache["list_flairs"]:
            return list(self._cache["list_flairs"][community_id])

        response = await self.client.get(f"{self.base_url}{self.community_prefix}/communities/{community_id}/flairs")
        response.raise_for_status()
        payload = response.json()
        self._cache["list_flairs"][community_id] = list(payload)
        return list(payload)

    async def get_post_comments(self, post_id: int, user_id: int | None = None) -> list[dict[str, Any]]:
        cache_key = (post_id, user_id)
        if cache_key in self._cache["get_post_comments"]:
            return list(self._cache["get_post_comments"][cache_key])

        response = await self.client.get(
            f"{self.base_url}{self.community_prefix}/posts/{post_id}/comments",
            headers=self._headers(user_id=user_id),
        )
        response.raise_for_status()
        payload = response.json()
        self._cache["get_post_comments"][cache_key] = list(payload)
        return list(payload)

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

    async def get_user_posts(
        self,
        username: str,
        user_id: int | None = None,
        limit: int = 25,
        day: int | None = None,
        month: int | None = None,
        year: int | None = None,
    ) -> list[dict[str, Any]]:
        query = (username or "").strip()
        if not query:
            return []

        posts = await self.search_posts(query, user_id=user_id, limit=max(10, min(limit, 60)))
        lowered = query.lower()
        filtered = [post for post in posts if lowered in str(post.get("authorName", "")).strip().lower()]

        if day is not None and month is not None and year is not None:
            filtered = [post for post in filtered if self._matches_date(post.get("createdAt"), day=day, month=month, year=year)]
        elif month is not None and year is not None:
            filtered = [post for post in filtered if self._matches_date(post.get("createdAt"), month=month, year=year)]
        elif year is not None:
            filtered = [post for post in filtered if self._matches_date(post.get("createdAt"), year=year)]

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
        results: list[dict[str, Any]] = []
        query_lower = query.lower()
        semaphore = asyncio.Semaphore(4)

        async def load_flairs(community: dict[str, Any]) -> list[dict[str, Any]]:
            community_id = community.get("id")
            if not community_id:
                return []
            async with semaphore:
                flairs = await self.list_flairs(int(community_id))
            matches: list[dict[str, Any]] = []
            for flair in flairs:
                flair_name = str(flair.get("name", "")).lower()
                if query_lower in flair_name:
                    matches.append(
                        {
                            "type": "flair",
                            "id": flair.get("id"),
                            "name": flair.get("name"),
                            "community_id": community_id,
                            "community_name": community.get("name"),
                            "color": flair.get("color"),
                            "textColor": flair.get("textColor"),
                        }
                    )
            return matches

        tasks = [load_flairs(community) for community in communities[:10]]
        for outcome in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(outcome, Exception):
                continue
            results.extend(outcome)
        return results

    async def search_rules_by_content(self, query: str, communities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        query_lower = query.lower()
        semaphore = asyncio.Semaphore(4)

        async def load_rules(community: dict[str, Any]) -> list[dict[str, Any]]:
            community_id = community.get("id")
            if not community_id:
                return []
            async with semaphore:
                rules = await self.get_community_rules(int(community_id))
            matches: list[dict[str, Any]] = []
            for rule in rules:
                rule_title = str(rule.get("title", "")).lower()
                rule_description = str(rule.get("description", "")).lower()
                if query_lower in rule_title or query_lower in rule_description:
                    matches.append(
                        {
                            "type": "rule",
                            "id": rule.get("id"),
                            "title": rule.get("title"),
                            "description": rule.get("description"),
                            "community_id": community_id,
                            "community_name": community.get("name"),
                            "order": rule.get("order"),
                        }
                    )
            return matches

        tasks = [load_rules(community) for community in communities[:10]]
        for outcome in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(outcome, Exception):
                continue
            results.extend(outcome)
        return results

    def _matches_date(
        self,
        value: Any,
        *,
        day: int | None = None,
        month: int | None = None,
        year: int | None = None,
    ) -> bool:
        parsed = self._parse_dt(value)
        if parsed is None:
            return False
        if year is not None and parsed.year != year:
            return False
        if month is not None and parsed.month != month:
            return False
        if day is not None and parsed.day != day:
            return False
        return True

    def _parse_dt(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None
