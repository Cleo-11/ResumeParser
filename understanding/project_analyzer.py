"""
understanding/project_analyzer.py

Layer 2 — Understanding
═══════════════════════
Responsibility: Deep-read each RawRepo and extract structured intelligence.
Input:          list[RawRepo]
Output:         list[ProjectUnderstanding]

Uses GPT-5 with structured JSON output to extract:
  - Project purpose (what & why)
  - Full tech stack (languages + frameworks + tools)
  - Complexity signal (1–5 score with rationale)
  - Impact signals (stars, CI, contributors, commit cadence)
  - Domain classification
  - Resume-optimised keywords
"""

from __future__ import annotations

from resume_pipeline.core.llm import LLMClient
from resume_pipeline.core.logger import LayerLogger
from resume_pipeline.core.models import RawRepo, ProjectUnderstanding

_log = LayerLogger("understanding")

SYSTEM = """You are a senior software engineer and technical recruiter hybrid.
Your job is to deeply analyse a GitHub project and extract structured intelligence
that will be used to write resume bullet points.

Be precise. If something cannot be inferred from the data, say so — don't hallucinate.
Return only valid JSON."""


def understand(repos: list[RawRepo], llm: LLMClient) -> list[ProjectUnderstanding]:
    """
    Run understanding analysis on every ingested repo.
    Returns one ProjectUnderstanding per repo.
    """
    _log.info(f"Starting understanding layer for {len(repos)} repo(s)")
    results: list[ProjectUnderstanding] = []

    for repo in repos:
        _log.debug(f"Analysing {repo.name} …")
        understanding = _analyse_repo(repo, llm)
        results.append(understanding)
        _log.success(f"{repo.name} → domain={understanding.domain}, complexity={understanding.complexity_score}/5")

    _log.success(f"Understanding complete — {len(results)} project(s) analysed", elapsed=_log.elapsed())
    return results


def _analyse_repo(repo: RawRepo, llm: LLMClient) -> ProjectUnderstanding:
    # Build a rich context payload for the LLM
    lang_breakdown = ", ".join(
        f"{lang} ({bytes_:,} bytes)" for lang, bytes_ in repo.languages_breakdown.items()
    ) or "Unknown"

    user_prompt = f"""
## Project: {repo.name}
**Description**: {repo.description or "None provided"}
**URL**: {repo.url}
**Primary language**: {repo.language or "Unknown"}
**Language breakdown**: {lang_breakdown}
**Topics/tags**: {", ".join(repo.topics) or "none"}
**Stars**: {repo.stars} | **Forks**: {repo.forks} | **Contributors**: {repo.contributors_count}
**Has test suite**: {repo.has_tests} | **Has CI/CD**: {repo.has_ci}
**License**: {repo.license_name or "None"}
**Created**: {repo.created_at.strftime("%Y-%m")} | **Last pushed**: {repo.pushed_at.strftime("%Y-%m")}

## Recent commit messages (last 20)
{chr(10).join(f"- {c}" for c in repo.recent_commits[:20]) or "No commits available"}

## README (first 3000 chars)
{repo.readme[:3000] or "No README available"}

---
Analyse this project and return a JSON object with these exact keys:
{{
  "repo_name": "{repo.name}",
  "project_purpose": "...",
  "tech_stack": ["...", "..."],
  "primary_language": "...",
  "complexity_score": 1-5,
  "complexity_rationale": "...",
  "impact_signals": ["...", "..."],
  "problem_solved": "...",
  "notable_features": ["...", "..."],
  "domain": "web|ml|devtools|mobile|infra|data|game|cli|library|other",
  "is_original_work": true|false,
  "keywords": ["...", "..."]
}}
"""
    result = llm.structured(SYSTEM, user_prompt, ProjectUnderstanding)
    # Inject dates from the RawRepo — the LLM doesn't need to produce these
    return result.model_copy(update={"pushed_at": repo.pushed_at, "created_at": repo.created_at})