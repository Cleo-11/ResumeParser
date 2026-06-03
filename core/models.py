"""
core/models.py
Shared Pydantic models that flow through every pipeline layer.
Each layer consumes the previous layer's output model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Layer 1 — Ingestion output
# ─────────────────────────────────────────────

class RawRepo(BaseModel):
    """Raw data pulled straight from GitHub APIs."""
    name: str
    full_name: str
    description: Optional[str] = None
    url: str
    language: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    pushed_at: datetime
    created_at: datetime
    readme: str = ""
    recent_commits: list[str] = Field(default_factory=list)   # commit messages
    languages_breakdown: dict[str, int] = Field(default_factory=dict)  # lang → bytes
    contributors_count: int = 0
    has_tests: bool = False
    has_ci: bool = False
    license_name: Optional[str] = None


# ─────────────────────────────────────────────
# Layer 2 — Understanding output
# ─────────────────────────────────────────────

class ProjectUnderstanding(BaseModel):
    """Structured understanding extracted by the LLM."""
    repo_name: str
    project_purpose: str = Field(description="What the project does and why it exists")
    tech_stack: list[str] = Field(description="All technologies, frameworks, libraries used")
    primary_language: str
    complexity_score: int = Field(ge=1, le=5, description="1=simple script, 5=complex system")
    complexity_rationale: str
    impact_signals: list[str] = Field(description="Evidence of real-world impact or usage")
    problem_solved: str = Field(description="The core problem this project addresses")
    notable_features: list[str] = Field(description="Stand-out technical or product features")
    domain: str = Field(description="e.g. web, ml, devtools, mobile, infra, data, etc.")
    is_original_work: bool = Field(description="True if original project, False if fork/clone")
    keywords: list[str] = Field(description="Resume-friendly keywords from the project")
    # Injected from RawRepo — not produced by the LLM
    pushed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


# ─────────────────────────────────────────────
# Layer 3 — Transformation output
# ─────────────────────────────────────────────

class ResumeBullet(BaseModel):
    """A single, polished resume bullet point with metadata."""
    text: str = Field(description="The bullet point (starts with action verb, ≤120 chars)")
    action_verb: str
    tech_mentioned: list[str]
    has_quantified_impact: bool
    confidence: float = Field(ge=0.0, le=1.0, description="How confident we are in quality")


class TransformedProject(BaseModel):
    """Resume-ready content produced by the transformation layer."""
    repo_name: str
    primary_bullet: ResumeBullet
    alternative_bullets: list[ResumeBullet] = Field(default_factory=list)
    skills_to_add: list[str] = Field(description="New skills to surface in skills section")
    section_target: str = Field(
        description="Which resume section this belongs to: Projects | Experience | Other"
    )
    suggested_title: str = Field(description="Project title for the resume entry")
    date_range: Optional[str] = None


# ─────────────────────────────────────────────
# Layer 4 — Editing output
# ─────────────────────────────────────────────

class EditDecision(BaseModel):
    """What the editing layer decided to do with a project."""
    repo_name: str
    action: str = Field(description="added | skipped_duplicate | skipped_low_quality | added_skills_only")
    reason: str
    bullet_used: Optional[str] = None
    skills_added: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Final result returned after the full pipeline completes."""
    output_path: str
    repos_ingested: int
    repos_processed: int
    decisions: list[EditDecision]
    total_bullets_added: int
    total_skills_added: int
    duration_seconds: float
    warnings: list[str] = Field(default_factory=list)