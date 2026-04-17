from __future__ import annotations

import json
from typing import Any


def build_planner_prompt(
    query: str,
    seed_posts: list[dict[str, Any]],
    communities: list[dict[str, Any]],
    action_budget: int,
) -> str:
    return (
        "You are an execution planner for community search.\n"
        "Return ONLY JSON with schema:\n"
        "{"
        '"normalized_query":"...",'
        '"intent":"...",'
        '"actions":[{"type":"search_posts|search_comments|get_user_posts|get_user_comments|get_post_by_id|get_related_posts|get_flair_trends|compare_communities|rank_results|summarize_with_citations|extract_actionable_advice|search_flairs|search_rules|list_communities|get_community_flairs|get_post_comments|get_trending_posts","args":{},"reason":"..."}]'
        "}\n"
        f"Max actions: {action_budget}.\n"
        "Prioritize multi-hop actions when user asks for analysis, comparison, recommendation, or explanation.\n"
        "Do not exceed max actions.\n\n"
        f"User query: {query}\n\n"
        f"Seed posts sample: {json.dumps(seed_posts[:5], ensure_ascii=False)}\n"
        f"Communities sample: {json.dumps(communities[:8], ensure_ascii=False)}"
    )


def build_synthesis_prompt(
    query: str,
    normalized_query: str,
    context: dict[str, Any],
    query_profile: dict[str, Any],
) -> str:
    return (
        "You are a community assistant. Use only the provided evidence.\n"
        "Return ONLY JSON with schema:\n"
        '{"answer":"...", "follow_ups":["...","...","..."], "confidence":0.0, "gaps":["..."]}\n'
        "Rules:\n"
        "- answer: 3-6 sentences, practical, contextual.\n"
        "- Include comparative or recommendation framing when the user intent asks for it.\n"
        "- Mention what drove ranking decisions when recommendation_mode=true.\n"
        "- If evidence is weak, list concrete gaps.\n"
        "- Never invent posts, comments, or metrics.\n\n"
        f"User query: {query}\n"
        f"Normalized query: {normalized_query}\n\n"
        f"Query profile: {json.dumps(query_profile, ensure_ascii=False)}\n"
        f"Posts: {json.dumps(context['trimmed_posts'], ensure_ascii=False)}\n"
        f"Communities: {json.dumps(context['trimmed_communities'], ensure_ascii=False)}\n"
        f"Comments: {json.dumps(context['trimmed_comments'], ensure_ascii=False)}\n"
        f"Flairs: {json.dumps(context['trimmed_flairs'], ensure_ascii=False)}\n"
        f"Rules: {json.dumps(context['trimmed_rules'], ensure_ascii=False)}\n"
        f"Ranking factors: {json.dumps(context['ranking_factors'][:6], ensure_ascii=False)}\n"
        f"Analysis notes: {json.dumps(context['analysis_notes'][:4], ensure_ascii=False)}\n"
        f"Actionable advice: {json.dumps(context['actionable_advice'][:5], ensure_ascii=False)}\n"
        f"Users in scope: {json.dumps(context['user_context'], ensure_ascii=False)}\n"
        f"Comments by post: {json.dumps(context['comments_by_post'], ensure_ascii=False)}\n"
    )


def build_query_profile_prompt(query: str) -> str:
    return (
        "Analyze this community search query and extract key information.\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{"
        '"author_target": null or "username",'
        '"animals": [] or ["dog", "cat"],'
        '"search_query": "what to actually search for",'
        '"is_author_query": true/false,'
        '"is_animal_query": true/false,'
        '"result_type": "posts" or "communities" or "flairs" or "rules" or "all",'
        '"freshness_days": null or integer,'
        '"comparison_confidence": 0.0,'
        '"recommendation_confidence": 0.0,'
        '"comparison_mode": true/false,'
        '"recommendation_mode": true/false,'
        '"explainability_mode": true/false'
        "}\n\n"
        "Rules:\n"
        "- author_target: extract username for 'posts by/from someone'.\n"
        "- animals: dog, cat, bird, fish, rabbit, hamster, guinea_pig, parrot.\n"
        "- freshness_days: parse phrases like 'last 7 days' or 'past month' into days.\n"
        "- comparison_mode: true when the user asks to compare options.\n"
        "- recommendation_mode: true when the user asks for best, top, or ranked picks.\n"
        "- explainability_mode: true when the user asks why or reasons, or when recommendations are requested.\n"
        "- result_type defaults to 'all' when unclear.\n"
        "- Confidence fields must be numbers between 0.0 and 1.0.\n\n"
        f"User query: {query}"
    )
