# Community AI Agent (Groq + FastAPI)

This is a standalone Python microservice that runs natural-language community search as an **agent loop**. It is used by the Elif application at https://github.com/AtfastrSlushyMaker/Elif.

1. Retrieves initial context from Elif community APIs.
2. Asks Groq to produce an action plan (`search_posts`, `get_post_comments`, `get_community_flairs`, etc.).
3. Executes those actions against backend APIs.
4. Asks Groq for a grounded final answer with follow-up questions.

## What this gives you

- Contextual answers that use posts + comments + communities + flairs.
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

Example request:

```json
{
  "query": "best pet transit checklist for EU",
  "user_id": 12,
  "include_trace": true
}
```

## Environment variables

Copy `.env.example` to `.env` and fill values.

- `GROQ_API_KEY` (required)
- `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)
- `BACKEND_BASE_URL` (default: `http://host.docker.internal:8087/elif`)
- `BACKEND_COMMUNITY_PREFIX` (default: `/api/community`)
- `HTTP_TIMEOUT_SECONDS` (default: `15`)
- `MAX_AGENT_ACTIONS` (default: `6`)

## Run locally

```bash
cd community-ai-agent-nl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8095 --reload
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

```bash
cd community-ai-agent-nl
docker build -t community-ai-agent .
docker run --env-file .env -p 8095:8095 community-ai-agent
```

## Development docs

- `ARCHITECTURE.md`
- `SCOPED_SEARCH.md`
- `ENTITIES_EXPANSION.md`

## Notes

- This service never hardcodes secrets. Keep API keys in `.env` or your secret manager.
- It is designed as an external service so you can deploy/version it independently from Elif.
