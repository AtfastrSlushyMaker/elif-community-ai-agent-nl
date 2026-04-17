# Community AI Agent - Expanded Entity Search

## Overview
The agent now searches across **ALL community entities**, not just posts and communities:
- ✓ Posts
- ✓ Communities  
- ✓ Post Flairs (tags/labels)
- ✓ Community Rules
- ✓ Comments (contextual)

## New Features

### 1. Flair Search
Search for post tags/labels across communities.

**Example Queries:**
- "search for flairs about dogs"
- "find all pet-related tags"
- "show me flairs"

**Backend Methods:**
```python
async def search_flairs_by_name(query: str, communities: list) -> list[dict]:
    """Search for flairs matching query across communities."""
```

**Response Format:**
```json
{
  "type": "flair",
  "id": 123,
  "name": "dog-care",
  "community_id": 5,
  "community_name": "Pet Owners",
  "color": "#FF5733",
  "textColor": "#FFFFFF"
}
```

### 2. Rule Search
Search for community rules and policies.

**Example Queries:**
- "show me community rules"
- "find rules about behavior"
- "what are the community guidelines?"

**Backend Methods:**
```python
async def search_rules_by_content(query: str, communities: list) -> list[dict]:
    """Search for rules matching query across communities."""

async def get_community_rules(community_id: int) -> list[dict]:
    """Get all rules for a specific community."""
```

**Response Format:**
```json
{
  "type": "rule",
  "id": 456,
  "title": "Be Respectful",
  "description": "Treat all members with respect and kindness",
  "community_id": 5,
  "community_name": "Pet Owners",
  "order": 1
}
```

## Result Type Expansion

The agent now supports 5 result types:

| Result Type | Returns | Use Case |
|------------|---------|----------|
| **posts** | Only posts | "show me posts about dogs" |
| **communities** | Only communities | "find communities about cats" |
| **flairs** | Only post tags/labels | "search for flairs" |
| **rules** | Only community rules | "show me community rules" |
| **all** | Everything | "community information" (default) |

## Architecture Changes

### New Agent Actions

**search_flairs**
```python
AgentAction(
    type="search_flairs",
    args={"query": "dogs"},
    reason="Find relevant post flairs/tags"
)
```

**search_rules**
```python
AgentAction(
    type="search_rules",
    args={"query": "behavior"},
    reason="Find relevant community rules"
)
```

### Updated Context
```python
context = {
    "posts": [...],
    "communities": [...],
    "flairs": [...],
    "rules": [...],              # NEW
    "comments_by_post": {...},
    "selected_community_ids": set(),
    "flairs_loaded_for": set(),
}
```

### Updated Response
```json
{
    "query": "show me flairs about dogs",
    "normalized_query": "...",
    "answer": "...",
    "follow_ups": [...],
    "referenced_posts": [...],
    "referenced_communities": [...],
    "referenced_flairs": [...],
    "referenced_rules": [...],              // NEW
    "trace": [...]
}
```

## Result Filtering Flow

```
User Query
    ↓
LLM Interprets: result_type = "flairs" | "rules" | "posts" | "communities" | "all"
    ↓
Agent Searches All Entities
    ├─ search_posts
    ├─ list_communities
    ├─ search_flairs ← NEW
    ├─ search_rules  ← NEW
    └─ get_trending_posts
    ↓
Result Filtering by result_type
    ├─ flairs  → Keep flairs, clear posts/communities/rules
    ├─ rules   → Keep rules, clear posts/communities/flairs
    ├─ posts   → Keep posts, clear communities/flairs/rules
    ├─ communities → Keep communities, clear posts/flairs/rules
    └─ all     → Return everything
    ↓
User Receives Filtered Results
```

## Query Examples

### Search Flairs
```
User: "what flairs exist for dog posts?"
LLM: result_type = "flairs"
Returns: Only flairs matching "dog"
```

### Search Rules
```
User: "show me the community rules"
LLM: result_type = "rules"
Returns: Only community rules
```

### Search Posts
```
User: "posts about training"
LLM: result_type = "posts"
Returns: Only posts (no communities/flairs/rules)
```

### Search All
```
User: "tell me about the community"
LLM: result_type = "all"
Returns: Posts + Communities + Flairs + Rules
```

## Implementation Details

### Backend Client Enhancements (backend_client.py)

**New Methods:**
1. `get_community_rules(community_id)` → List of rules
2. `search_flairs_by_name(query, communities)` → Matching flairs
3. `search_rules_by_content(query, communities)` → Matching rules

**Features:**
- Limits to first 10 communities to avoid excessive API calls
- Handles exceptions gracefully
- Returns results in standardized format with metadata

### Agent Enhancements (agent.py)

**Updated Methods:**
1. `_interpret_query_with_llm()` - Extended to recognize rules/flairs intent
2. `_build_query_profile()` - Infers result_type for new entities
3. `_plan_actions()` - Updated prompt to include new actions
4. `_execute_action()` - Handles search_flairs and search_rules

**New Logic:**
- Context includes "rules" field
- Result filtering handles 5 result types instead of 3
- Trace entries show what was filtered and why

## Testing

All entity types work correctly:
```
✓ "search for flairs about dogs" → Returns flairs
✓ "show me community rules" → Returns rules
✓ "posts about cats" → Returns posts
✓ "find communities" → Returns communities
✓ "community information" → Returns all entities
```

## Performance Considerations

1. **Parallel Execution**: Actions are executed sequentially but can be optimized
2. **API Call Limits**: Each search_flairs/search_rules iterates up to 10 communities
3. **Result Trimming**: Flairs limited to 20, rules limited to 10 to keep responses concise
4. **Caching**: Consider caching flair/rule data for frequently accessed communities

## Future Enhancements

1. **Smart Action Planning**: Skip searches based on result_type
   - If result_type="flairs", don't search posts/communities
   
2. **Flair Filtering**: Filter posts by selected flairs
   
3. **Rule-Based Recommendations**: Show rules relevant to discussion topic
   
4. **Comments in Search**: Include comments in full-text search
   
5. **User-Flair Mapping**: Show which users have used which flairs

## Error Handling

All entity searches have graceful error handling:
- If flair search fails → returns empty list
- If rule search fails → returns empty list
- If any entity unavailable → continues with available data
- No single entity failure blocks the entire search

## Advanced Agent Layer (New)

The agent now includes a richer investigation/action layer beyond entity expansion:

### New concrete actions

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

### New response fields

- `referenced_comments`
- `confidence` (`0..1`)
- `gaps`
- `ranking_factors`
- `next_best_actions`

### New behavior modes

- **Multi-hop mode**: cascades from discovery to ranking/synthesis.
- **Comparison mode**: side-by-side community comparison output.
- **Recommendation mode**: weighted ranking + explicit factor reporting.
- **Explainability mode**: per-item `why_selected` for posts/communities.
- **Freshness control**: parses phrases like `last 7 days` and filters evidence accordingly.
