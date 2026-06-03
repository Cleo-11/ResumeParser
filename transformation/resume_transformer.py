"""
transformation/resume_transformer.py

Layer 3 — Transformation
═════════════════════════
Responsibility: Convert structured understanding into polished resume content.
Input:          list[ProjectUnderstanding]
Output:         list[TransformedProject]

This layer applies resume-writing craft:
  - Strong past-tense action verbs (Built, Developed, Engineered, Architected…)
  - Quantified impact wherever signals exist (stars, contributors, LOC, etc.)
  - Three alternative bullets at different emphasis angles
  - Section targeting (Projects vs Experience)
  - Skills list de-duplication
"""

from __future__ import annotations
import json

from core.llm import LLMClient
from core.logger import LayerLogger
from core.models import (
    ProjectUnderstanding,
    TransformedProject,
    ResumeBullet,
)

_log = LayerLogger("transformation")

# Strong action verbs pool — categorised by domain
ACTION_VERBS = {
    "web":      ["Architected", "Engineered", "Built", "Developed", "Launched"],
    "ml":       ["Trained", "Implemented", "Developed", "Designed", "Optimised"],
    "devtools": ["Built", "Shipped", "Authored", "Engineered", "Published"],
    "infra":    ["Deployed", "Automated", "Orchestrated", "Provisioned", "Scaled"],
    "data":     ["Designed", "Built", "Developed", "Integrated", "Optimised"],
    "library":  ["Published", "Open-sourced", "Authored", "Maintained", "Shipped"],
    "cli":      ["Built", "Shipped", "Authored", "Created", "Developed"],
    "other":    ["Built", "Developed", "Engineered", "Created", "Designed"],
}

SYSTEM = """You are an elite resume writer with 15 years of experience helping
engineers land roles at top-tier companies.

Rules for bullet points:
1. Start with a strong past-tense action verb (never "worked on", "helped with")
2. Be specific — name the exact technologies used
3. Quantify impact wherever the data supports it (stars, users, % improvement, LOC, etc.)
4. Keep each bullet under 120 characters
5. Never use first-person pronouns
6. No fluff ("leveraged", "utilized", "facilitated")

Return ONLY valid JSON."""


def transform(
    understandings: list[ProjectUnderstanding],
    llm: LLMClient,
) -> list[TransformedProject]:
    """Convert all project understandings to resume-ready content."""
    _log.info(f"Starting transformation for {len(understandings)} project(s)")
    results: list[TransformedProject] = []

    for u in understandings:
        _log.debug(f"Transforming {u.repo_name} (domain={u.domain}, complexity={u.complexity_score})")
        transformed = _transform_one(u, llm)
        results.append(transformed)
        _log.success(f"{u.repo_name} → '{transformed.primary_bullet.text[:60]}…'")

    _log.success(f"Transformation complete — {len(results)} project(s) transformed", elapsed=_log.elapsed())
    return results


def _transform_one(u: ProjectUnderstanding, llm: LLMClient) -> TransformedProject:
    verbs = ACTION_VERBS.get(u.domain, ACTION_VERBS["other"])
    verb_hint = ", ".join(verbs)

    # Quantification hints from ingestion signals
    quant_hints = []
    for signal in u.impact_signals:
        quant_hints.append(signal)

    user_prompt = f"""
## Project to write bullets for

**Name**: {u.repo_name}
**Purpose**: {u.project_purpose}
**Problem solved**: {u.problem_solved}
**Tech stack**: {", ".join(u.tech_stack)}
**Domain**: {u.domain}
**Complexity**: {u.complexity_score}/5 — {u.complexity_rationale}
**Notable features**: {", ".join(u.notable_features)}
**Impact signals**: {", ".join(u.impact_signals) or "none quantifiable"}
**Keywords**: {", ".join(u.keywords)}

Suggested action verbs for this domain: {verb_hint}

Write:
1. ONE primary bullet point (the best possible, under 120 chars)
2. TWO alternative bullets with different emphasis angles (one tech-focused, one impact-focused)
3. A list of skills/technologies to add to the Skills section (short, clean terms like "React", "PostgreSQL", "Docker")
4. The best section target: "Projects" or "Experience"
5. A concise title for this resume entry (e.g. "Real-Time Chat App", "ML Image Classifier CLI")

Return JSON:
{{
  "primary_bullet": {{
    "text": "...",
    "action_verb": "...",
    "tech_mentioned": ["..."],
    "has_quantified_impact": true/false,
    "confidence": 0.0-1.0
  }},
  "alternative_bullets": [
    {{ "text": "...", "action_verb": "...", "tech_mentioned": [...], "has_quantified_impact": ..., "confidence": ... }},
    {{ "text": "...", "action_verb": "...", "tech_mentioned": [...], "has_quantified_impact": ..., "confidence": ... }}
  ],
  "skills_to_add": ["...", "..."],
  "section_target": "Projects",
  "suggested_title": "..."
}}
"""
    raw = llm.structured(SYSTEM, user_prompt, _TransformResponse)
    return TransformedProject(
        repo_name=u.repo_name,
        primary_bullet=raw.primary_bullet,
        alternative_bullets=raw.alternative_bullets,
        skills_to_add=raw.skills_to_add,
        section_target=raw.section_target,
        suggested_title=raw.suggested_title,
        date_range=_compute_date_range(u),
    )


# Internal response schema for the LLM structured output
from pydantic import BaseModel, Field
from typing import Optional

class _TransformResponse(BaseModel):
    primary_bullet: ResumeBullet
    alternative_bullets: list[ResumeBullet] = Field(default_factory=list)
    skills_to_add: list[str] = Field(default_factory=list)
    section_target: str = "Projects"
    suggested_title: str = ""


def _compute_date_range(u: ProjectUnderstanding) -> Optional[str]:
    """Build a human-readable date range from actual GitHub push/creation dates."""
    if not u.pushed_at:
        return None
    start = u.created_at or u.pushed_at
    start_str = start.strftime("%b %Y")
    end_str = u.pushed_at.strftime("%b %Y")
    if start_str == end_str:
        return start_str
    return f"{start_str} – {end_str}"