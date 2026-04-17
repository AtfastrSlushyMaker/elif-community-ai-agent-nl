from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import CommunitySearchAgent
from .backend_client import BackendClient
from .config import settings
from .models import AgentSearchRequest, AgentSearchResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = BackendClient(
        base_url=settings.backend_base_url,
        community_prefix=settings.backend_community_prefix,
        timeout_seconds=settings.http_timeout_seconds,
    )
    agent = CommunitySearchAgent(
        backend=backend,
        groq_api_key=settings.groq_api_key,
        groq_model=settings.groq_model,
        max_actions=settings.max_agent_actions,
    )
    app.state.backend = backend
    app.state.agent = agent
    try:
        yield
    finally:
        await backend.close()


app = FastAPI(title="Community AI Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/community/agent-search", response_model=AgentSearchResponse)
async def agent_search(payload: AgentSearchRequest) -> AgentSearchResponse:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    import traceback
    try:
        result = await app.state.agent.run(
            query=payload.query,
            user_id=payload.user_id,
            act_as_user_id=payload.act_as_user_id,
            max_actions=payload.max_actions,
            community_id=payload.community_id,
        )
    except Exception as ex:
        print("\n--- AGENT EXECUTION ERROR ---")
        traceback.print_exc()
        print("--- END ERROR ---\n")
        raise HTTPException(status_code=502, detail=f"Agent execution failed: {ex}") from ex

    if not payload.include_trace:
        result["trace"] = None
    return AgentSearchResponse(**result)
