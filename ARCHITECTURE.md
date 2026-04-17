# Community AI Agent - Architecture & Feature Modes

## Overview

The agent runs a plan-and-execute loop with strict action contracts. The LLM proposes actions; Python executes them against backend APIs and deterministic ranking/comparison logic.

## End-to-end flow

1. **Interpret query intent**
   - Extracts author target, animals, result type, freshness window, and behavior modes.
   - Modes include:
     - `multi_hop_mode`
     - `comparison_mode`
     - `recommendation_mode`
     - `explainability_mode`

2. **Seed evidence**
   - Pulls initial posts and communities.
   - Applies scoped search (`community_id`) and freshness filtering when requested.

3. **Plan actions**
   - LLM returns typed JSON action list.
   - Agent augments plan based on intent (author, comparison, recommendation, explainability).

4. **Execute actions safely**
   - New action families:
     - **Discovery**: `search_comments`, `get_user_posts`, `get_user_comments`, `get_post_by_id`, `get_related_posts`
     - **Analytics**: `get_flair_trends`, `compare_communities`, `rank_results`
     - **Synthesis support**: `summarize_with_citations`, `extract_actionable_advice`

5. **Synthesize grounded answer**
   - Returns answer + follow-ups + confidence + gaps.
   - Includes explainability fields (`why_selected`) and ranking factors.

## Result contract

The agent now returns:

- `referenced_posts`
- `referenced_communities`
- `referenced_comments`
- `referenced_flairs`
- `referenced_rules`
- `confidence` (`0..1`)
- `gaps`
- `ranking_factors`
- `next_best_actions`

## Behavior details

- **Multi-hop mode**: chains actions to gather and refine evidence before synthesis.
- **Comparison mode**: builds side-by-side community comparisons from activity and engagement signals.
- **Recommendation mode**: ranks posts/communities with explicit weighted factors.
- **Explainability mode**: each referenced post/community can include `why_selected`.
- **Freshness control**: phrases like `last 7 days`, `past month`, `last week` are converted to date windows and enforced in evidence filtering/ranking.
