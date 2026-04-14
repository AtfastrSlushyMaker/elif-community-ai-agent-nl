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
