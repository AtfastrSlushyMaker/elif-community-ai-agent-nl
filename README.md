# Community AI Agent (LLM + FastAPI)

This is a standalone Python microservice that runs natural-language community search as an **agent loop**. It is used by the Elif application at https://github.com/AtfastrSlushyMaker/Elif.

1. Retrieves initial context from Elif community APIs.
2. Asks an LLM to produce an action plan (`search_posts`, `get_post_comments`, `get_community_flairs`, etc.).
3. Executes those actions against backend APIs.
4. Asks the LLM for a grounded final answer with follow-up questions.

In this instance, the implementation uses Groq, but the agent pattern is provider-agnostic and can be adapted to your preferred LLM backend.

## How the agent works

The service follows a plan-and-execute loop with guardrails:

1. Build a query profile from the user question (author intent, animal/species intent, topic tokens).
2. Fetch seed evidence from backend APIs (`search_posts`, `list_communities`).
3. Ask the configured LLM to return a strict JSON action plan.
4. Execute actions safely in Python (not directly by the model).
5. Auto-enrich context with flairs and comments.
6. Ask the configured LLM for a grounded final answer using only collected evidence.

### Runtime wiring (FastAPI lifespan)

At app startup, one backend client and one agent instance are created and stored in `app.state`:

```python
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
        groq_api_keys=settings.groq_api_keys,
        groq_model=settings.groq_model,
        max_actions=settings.max_agent_actions,
    )
    app.state.backend = backend
    app.state.agent = agent
    try:
        yield
    finally:
        await backend.close()
```

### API entrypoint

The endpoint validates input, runs the agent, and returns a typed response:

```python
@app.post("/v1/community/agent-search", response_model=AgentSearchResponse)
async def agent_search(payload: AgentSearchRequest) -> AgentSearchResponse:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    result = await app.state.agent.run(
        query=payload.query,
        user_id=payload.user_id,
        act_as_user_id=payload.act_as_user_id,
        max_actions=payload.max_actions,
        community_id=payload.community_id,
    )

    if not payload.include_trace:
        result["trace"] = None
    return AgentSearchResponse(**result)
```

### Core agent loop

`CommunitySearchAgent.run(...)` orchestrates planning, tool execution, enrichment, and synthesis:

```python
query_profile = self._build_query_profile(query)
seed_posts = await self.backend.search_posts(query, user_id=user_id, limit=12)
communities = await self.backend.list_communities(user_id=user_id)

plan = await self._plan_actions(
    query=query,
    seed_posts=seed_posts,
    communities=communities,
    action_budget=action_budget,
)

for action in plan.actions[:action_budget]:
    await self._execute_action(
        action,
        context=context,
        user_id=user_id,
        community_index=community_index,
        community_focus=community_focus,
        query_profile=query_profile,
    )

await self._ensure_comment_context(context=context, user_id=user_id, trace=trace)
synthesis = await self._synthesize(
    query=query,
    normalized_query=plan.normalized_query,
    context=context,
    query_profile=query_profile,
)
```

### Planner contract (JSON only)

The planner asks the configured LLM for strict JSON and only allows known action types:

```json
{
  "normalized_query": "cat travel checklist",
  "intent": "search",
  "actions": [
    {
      "type": "search_posts",
      "args": { "query": "cat travel checklist", "limit": 18 },
      "reason": "Collect relevant threads"
    },
    {
      "type": "list_communities",
      "args": { "query": "cat", "limit": 12 },
      "reason": "Find relevant communities"
    },
    {
      "type": "get_trending_posts",
      "args": { "sort": "HOT", "window": "WEEK", "limit": 8 },
      "reason": "Add high-signal context"
    }
  ]
}
```

Supported action types:

- `search_posts`
- `search_comments`
- `get_user_posts`
- `get_user_comments`
- `get_post_by_id`
- `get_related_posts`
- `get_flair_trends`
- `compare_communities`
- `rank_results`
- `summarize_with_citations`
- `extract_actionable_advice`
- `search_flairs`
- `search_rules`
- `list_communities`
- `get_community_flairs`
- `get_post_comments`
- `get_trending_posts`

If planner output is invalid or empty, the code falls back to a default deterministic plan.

### Example API call

```bash
curl -X POST http://127.0.0.1:8095/v1/community/agent-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best pet transit checklist for EU",
    "user_id": 12,
    "include_trace": true,
    "max_actions": 6
  }'
```

### Example response shape

```json
{
  "query": "best pet transit checklist for EU",
  "normalized_query": "pet transit checklist eu",
  "answer": "...grounded answer...",
  "follow_ups": [
    "Show me top posts about pet passports",
    "Which communities discuss EU relocation most?",
    "What common mistakes should I avoid?"
  ],
  "referenced_posts": [{"id": 123, "title": "..."}],
  "referenced_communities": [{"id": 9, "name": "..."}],
  "referenced_comments": [{"id": 98, "postId": 123, "authorName": "lina"}],
  "referenced_flairs": ["Travel", "Paperwork"],
  "referenced_rules": [{"id": 77, "title": "No spam"}],
  "confidence": 0.82,
  "gaps": ["Limited recent comments from moderators"],
  "ranking_factors": [{"factor": "balanced_ranking"}],
  "next_best_actions": ["compare_communities", "search_comments"],
  "model": "llama-3.3-70b-versatile",
  "trace": [
    {"step": "seed_fetch", "posts": 12, "communities": 18},
    {"step": "plan", "actions": ["..."]},
    {"step": "action", "result": {"ok": true}}
  ]
}
```

### Why this is reliable

- The model proposes actions, but Python executes them.
- Action types are constrained by typed schemas.
- Parsing is defensive (JSON extraction + fallback plan).
- Final synthesis is instructed to use only retrieved evidence.

## What this gives you

- Contextual answers that use posts + comments + communities + flairs + rules.
- Multi-hop behavior that chains evidence gathering (posts → comments → ranking/comparison → synthesis).
- Comparison mode with side-by-side community scoring when user intent asks to compare.
- Recommendation + explainability with `why_selected` on referenced posts and communities.
- Freshness control (e.g. "last 7 days") applied to evidence selection and ranking.
- Deterministic tool execution (the LLM proposes actions; Python executes safely).
- Portable service you can move to a dedicated repo and containerize.

## Endpoints

- `GET /health`
- `POST /v1/community/agent-search`

Request fields for `POST /v1/community/agent-search`:

- `query` (required)
- `user_id` (optional)
- `act_as_user_id` (optional)
- `include_trace` (optional, default `false`)
- `max_actions` (optional, `1..12`)
- `community_id` (optional, restricts search to one community)

Example request:

```json
{
  "query": "best pet transit checklist for EU",
  "user_id": 12,
  "include_trace": true
}
```

## Environment variables

Use one of the dedicated templates:

- Local runtime: `cp .env.local.example .env`
- Docker runtime: `cp .env.docker.example .env`

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEYS` | Recommended | `""` | Comma-separated list of Groq API keys. The agent tries each key in order; if one is rate-limited (429) it rotates to the next. |
| `GROQ_API_KEY` | Yes (if `GROQ_API_KEYS` is empty) | `""` | Single Groq API key used as fallback when `GROQ_API_KEYS` is not set. |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model identifier. |
| `LLM_PROVIDER` | No | `groq` | LLM provider name (reserved for future multi-provider support). |
| `BACKEND_BASE_URL` | Yes | `http://localhost:8087/elif` | Base URL of the Elif backend. Use `http://host.docker.internal:8087/elif` when running in Docker. |
| `BACKEND_COMMUNITY_PREFIX` | No | `/api/community` | API prefix for community endpoints. |
| `HTTP_TIMEOUT_SECONDS` | No | `30` | HTTP timeout for backend and Groq API calls. |
| `MAX_AGENT_ACTIONS` | No | `6` | Maximum number of agent actions per query. |

To use another LLM provider, replace the LLM client implementation in the agent while keeping the same planner/synthesis JSON contracts.

## Run locally

```bash
cd community-ai-agent-nl
cp .env.local.example .env
# Edit .env and add your Groq API key(s)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8095 --reload
```

Open `http://127.0.0.1:8095/health` to verify the service is running.

## Python version note

This project works with Python 3.14, but dependency resolution requires a modern pydantic stack.
The requirements use range pins for `pydantic` and `pydantic-settings` to ensure compatible
`pydantic-core` wheels are selected.

If setup fails in an older virtual environment, recreate it and reinstall:

```bash
rm -rf .venv
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Docker

### Build the image

```bash
cd community-ai-agent-nl
docker build -t community-ai-agent .
```

### Run the container

Set up your environment file first:

```bash
cp .env.docker.example .env
# Edit .env and add your Groq API key(s)
```

The run command differs slightly by operating system because of how Docker containers reach the host network.

#### macOS (Docker Desktop)

`host.docker.internal` is built into Docker Desktop — no extra flags needed:

```bash
docker run --env-file .env -p 8095:8095 community-ai-agent
```

#### Windows (Docker Desktop)

Same as macOS — `host.docker.internal` is built into Docker Desktop:

```powershell
docker run --env-file .env -p 8095:8095 community-ai-agent
```

#### Linux

On Linux, `host.docker.internal` does not exist by default. You must map it manually with `--add-host`:

```bash
docker run --add-host host.docker.internal:host-gateway --env-file .env -p 8095:8095 community-ai-agent
```

> **Note:** On Linux with Docker Engine (not Docker Desktop), if `host-gateway` is not supported on your Docker version (< 20.10), replace it with the host's actual IP address (e.g. `172.17.0.1` for the default Docker bridge).

### Verify

Once the container is running, check the health endpoint:

```bash
curl http://localhost:8095/health
# Expected: {"status":"ok"}
```

### Run with Docker Compose (optional)

If you prefer docker compose, create a `docker-compose.yml`:

```yaml
services:
  community-ai-agent:
    build: .
    env_file: .env
    ports:
      - "8095:8095"
    # Linux only — uncomment the next two lines:
    # extra_hosts:
    #   - "host.docker.internal:host-gateway"
```

Then run:

```bash
docker compose up --build
```

### Platform differences summary

| Feature | macOS | Windows | Linux |
|---|---|---|---|
| Docker Desktop | Yes | Yes | Optional |
| `host.docker.internal` built-in | ✅ | ✅ | ❌ |
| Extra flag needed | None | None | `--add-host host.docker.internal:host-gateway` |
| Docker Engine version | Any | Any | ≥ 20.10 for `host-gateway` |

## Backend contract check (agent actions)

The backend endpoints required by agent-executed API calls are available in Elif backend controllers:

- `GET /api/community/posts/search` (`PostController`)
- `GET /api/community/posts/trending` (`PostController`)
- `GET /api/community/posts/{id}` (`PostController`)
- `GET /api/community/posts/{postId}/comments` (`CommentController`)
- `GET /api/community/communities` (`CommunityController`)
- `GET /api/community/communities/{id}/posts` (`PostController`)
- `GET /api/community/communities/{id}/flairs` (`CommunityController`)
- `GET /api/community/communities/{id}/rules` (`CommunityController`)

## Development docs

- `ARCHITECTURE.md`
- `SCOPED_SEARCH.md`
- `ENTITIES_EXPANSION.md`

## Notes

- This service never hardcodes secrets. Keep API keys in `.env` or your secret manager.
- It is designed as an external service so you can deploy/version it independently from Elif.
- When using multiple `GROQ_API_KEYS`, the agent tries them sequentially on rate-limit errors (429) for automatic failover.
