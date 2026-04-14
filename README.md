# Community AI Agent (Groq + FastAPI)

This is a standalone Python microservice that runs natural-language community search as an **agent loop**:

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
cd community-ai-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8095 --reload
```

## Docker

```bash
cd community-ai-agent
docker build -t community-ai-agent .
docker run --env-file .env -p 8095:8095 community-ai-agent
```

## Notes

- This service never hardcodes secrets. Keep API keys in `.env` or your secret manager.
- It is designed as an external service so you can deploy/version it independently from Elif.
