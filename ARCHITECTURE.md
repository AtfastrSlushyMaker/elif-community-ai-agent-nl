# Community AI Agent - Architecture & Query Flow

## Overview
The agent now uses **LLM-powered intent understanding** to interpret user queries and filter results accordingly.

## Query Processing Pipeline

```
User Query (e.g., "communities about dogs")
    ↓
[1] _build_query_profile() - Interpret Intent
    ├─ Try: _interpret_query_with_llm()
    │   └─ Groq extracts: author, animals, result_type
    └─ Fallback: Regex extraction with inference
    ↓
[2] Query Profile Created
    {
        "author_target": null,
        "animals": ["dog"],
        "result_type": "communities",  ← KEY
        "strict_author": false,
        "strict_animals": true,
        "topic_tokens": [...],
        "llm_interpreted": true
    }
    ↓
[3] Search Execution (via agent loop)
    ├─ Search posts about dogs
    ├─ Search communities with dogs
    ├─ Load comments, flairs, etc.
    └─ Build context with BOTH posts and communities
    ↓
[4] Result Filtering (Final Step)
    ├─ Read result_type from query profile
    ├─ If result_type == "communities"
    │   └─ Clear referenced_posts = []
    └─ If result_type == "posts"
        └─ Clear referenced_communities = []
    ↓
[5] Return Results
    {
        "referenced_posts": [...],
        "referenced_communities": [...],  ← ONLY these returned
        "trace": [...]
    }
```

## Key Components

### 1. `_interpret_query_with_llm(query: str)` → dict
**Purpose:** Use Groq LLM to intelligently interpret user intent

**Input:** Raw user query string

**Output:** Structured dict with:
- `author_target`: Username if user wants posts from someone
- `animals`: List of pet types mentioned
- `search_query`: Core keywords cleaned of stopwords
- `is_author_query`: Boolean flag
- `is_animal_query`: Boolean flag
- **`result_type`: "posts" | "communities" | "both"** ← NEW
- `llm_interpreted`: Boolean

**Error Handling:** Returns `None` on failure → Fallback to regex

### 2. `_build_query_profile(query: str)` → dict
**Purpose:** Build comprehensive query profile for agent

**Flow:**
1. Try LLM interpretation first (better NLP)
2. Fallback to regex + inference (robustness)
3. Ensure result_type field always present

**Always Returns:**
- All fields from interpreter
- `result_type` field (from LLM or inferred)

### 3. `run()` → dict (Final Step)
**Purpose:** Execute search and filter results by intent

**Key Addition (lines 157-169):**
```python
# Filter results based on user's result_type intent
result_type = query_profile.get("result_type", "both")
referenced_posts = self._trim_posts(context["posts"])
referenced_communities = self._trim_communities(context["communities"])

if result_type == "posts":
    referenced_communities = []  # Clear communities
elif result_type == "communities":
    referenced_posts = []  # Clear posts
```

## Result Type Logic

### Groq LLM Rules (in prompt)
The LLM is instructed to classify queries:

```
"posts" if user says:
  - "posts about"
  - "show me posts"
  - "find posts"
  - "post from/by"

"communities" if user says:
  - "communities about"
  - "find communities"
  - "communities with"
  - "groups about"

"both" if:
  - Query is ambiguous
  - Doesn't specify type
  - User wants everything
```

### Fallback Inference Rules (if LLM fails)
Keyword-based classification:

```python
if "communities" in query or "groups" in query:
    result_type = "communities"
elif "posts" in query or "post from" in query:
    result_type = "posts"
else:
    result_type = "both"  # Default to both when unsure
```

## Example Flows

### Example 1: "communities about dogs"
```
1. Query Profile: {
     "result_type": "communities",
     "animals": ["dog"],
     "author_target": null
   }

2. Search Execution:
   - Searches for communities with "dog"
   - Also finds posts about dogs (contextually useful)
   - Loads both into context

3. Result Filtering:
   - result_type == "communities"
   - referenced_posts = []  ← Cleared
   - referenced_communities = [...]  ← Returned

4. User Gets: ONLY communities ✓
```

### Example 2: "posts from lina"
```
1. Query Profile: {
     "result_type": "posts",
     "author_target": "lina",
     "animals": []
   }

2. Search Execution:
   - Searches for "lina" posts
   - Searches general posts
   - Loads both into context

3. Result Filtering:
   - result_type == "posts"
   - referenced_communities = []  ← Cleared
   - referenced_posts = [...]  ← Returned

4. User Gets: ONLY posts from lina ✓
```

## Benefits of This Approach

✅ **Natural Language Aware**: LLM understands varied phrasings  
✅ **Explicit Intent**: User's desired result type is clear  
✅ **Simple Filtering**: Filter at the end, not throughout pipeline  
✅ **Robust Fallback**: Regex with smart inference if LLM fails  
✅ **Transparent**: Trace shows what result_type was selected  
✅ **No Over-engineering**: Uses existing infrastructure (Groq LLM)  
✅ **Backward Compatible**: Falls back gracefully if LLM unavailable

## Tracing & Debugging

The agent includes trace entries for transparency:

```python
trace.append({
    "step": "result_filtering",
    "reason": "result_type is communities",
    "cleared": "posts"
})
```

This shows in the response so users/developers can understand why certain results were filtered.

## Future Enhancements

1. **Action Planning Optimization**: Don't search for posts if only communities are wanted
2. **Confidence Scoring**: Add confidence level to LLM classification
3. **Query Caching**: Cache profiles for repeated queries
4. **A/B Testing**: Compare LLM vs regex classification accuracy
5. **User Feedback Loop**: Learn from corrections
