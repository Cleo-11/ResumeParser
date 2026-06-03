"""
pipeline.py
Main orchestrator — runs all 4 layers in sequence.
"""

from __future__ import annotations

# Load .env file AFTER the __future__ import, before everything else
import os
from pathlib import Path

# Support python-dotenv if installed, silently skip if not
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from resume_pipeline.core.llm import LLMClient
from resume_pipeline.core.logger import LayerLogger, add_handler
from resume_pipeline.core.models import PipelineResult, EditDecision
from resume_pipeline.ingestion.github_ingestor import ingest
from resume_pipeline.understanding.project_analyzer import understand
from resume_pipeline.transformation.resume_transformer import transform
from resume_pipeline.editing.resume_editor import edit

_log = LayerLogger("pipeline")


@dataclass
class PipelineConfig:
    github_username: str
    openai_api_key: str
    resume_path: str
    output_path: str = "resume_updated.docx"
    github_token: Optional[str] = None
    model: str = "gpt-4o"
    days_back: int = 90
    max_repos: int = 8
    include_forks: bool = False
    min_confidence: float = 0.6
    dry_run: bool = False
    event_handlers: List[Callable] = field(default_factory=list)


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        for handler in config.event_handlers:
            add_handler(handler)

    def run(self) -> PipelineResult:
        cfg = self.config
        start = time.perf_counter()
        warnings: List[str] = []

        _log.info("=" * 50)
        _log.info("  RESUME PIPELINE STARTING")
        _log.info("=" * 50)

        if not Path(cfg.resume_path).exists():
            raise FileNotFoundError(f"Resume not found: {cfg.resume_path}")

        _log.info(f"LLM: {cfg.model}")
        llm = LLMClient(api_key=cfg.openai_api_key, model=cfg.model)

        _log.info("LAYER 1 - Ingestion")
        raw_repos = ingest(
            username=cfg.github_username,
            token=cfg.github_token,
            days_back=cfg.days_back,
            max_repos=cfg.max_repos,
            include_forks=cfg.include_forks,
        )
        if not raw_repos:
            _log.warn("No repos found - nothing to process.")
            return PipelineResult(
                output_path=cfg.resume_path,
                repos_ingested=0, repos_processed=0,
                decisions=[], total_bullets_added=0, total_skills_added=0,
                duration_seconds=0, warnings=["No repos found in window"],
            )

        _log.info("LAYER 2 - Understanding")
        understandings = understand(raw_repos, llm)

        _log.info("LAYER 3 - Transformation")
        transformed = transform(understandings, llm)

        if cfg.dry_run:
            _log.info("LAYER 4 - Editing (DRY RUN - no file changes)")
            decisions: List[EditDecision] = [
                EditDecision(
                    repo_name=t.repo_name,
                    action="added",
                    reason="dry_run",
                    bullet_used=t.primary_bullet.text,
                    skills_added=t.skills_to_add,
                )
                for t in transformed
            ]
            out_path = cfg.resume_path
        else:
            _log.info("LAYER 4 - Editing")
            decisions = edit(
                projects=transformed,
                resume_path=cfg.resume_path,
                output_path=cfg.output_path,
                llm=llm,
                min_confidence=cfg.min_confidence,
            )
            out_path = cfg.output_path

        added = [d for d in decisions if d.action == "added"]
        total_skills = sum(len(d.skills_added) for d in added)
        duration = round(time.perf_counter() - start, 2)

        _log.info("=" * 50)
        _log.success(f"  PIPELINE COMPLETE in {duration}s")
        _log.success(f"  Repos scanned:   {len(raw_repos)}")
        _log.success(f"  Bullets added:   {len(added)}")
        _log.success(f"  Skills added:    {total_skills}")
        _log.success(f"  Token usage:     {llm.usage_summary}")
        _log.info("=" * 50)

        return PipelineResult(
            output_path=out_path,
            repos_ingested=len(raw_repos),
            repos_processed=len(transformed),
            decisions=decisions,
            total_bullets_added=len(added),
            total_skills_added=total_skills,
            duration_seconds=duration,
            warnings=warnings,
        )


def _cli():
    parser = argparse.ArgumentParser(description="Resume Pipeline")
    parser.add_argument("--github-user",    default=os.getenv("GITHUB_USERNAME"))
    parser.add_argument("--resume",         required=True)
    parser.add_argument("--output",         default="resume_updated.docx")
    parser.add_argument("--openai-key",     default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--github-token",   default=os.getenv("GITHUB_TOKEN"))
    parser.add_argument("--model",          default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--days-back",      type=int, default=90)
    parser.add_argument("--max-repos",      type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--include-forks",  action="store_true")
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--json-out",       default=None)

    args = parser.parse_args()

    if not args.github_user:
        parser.error("--github-user is required (or set GITHUB_USERNAME in .env)")
    if not args.openai_key:
        parser.error("--openai-key is required (or set OPENAI_API_KEY in .env)")

    config = PipelineConfig(
        github_username=args.github_user,
        openai_api_key=args.openai_key,
        resume_path=args.resume,
        output_path=args.output,
        github_token=args.github_token,
        model=args.model,
        days_back=args.days_back,
        max_repos=args.max_repos,
        include_forks=args.include_forks,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
    )

    result = Pipeline(config).run()

    if args.json_out:
        Path(args.json_out).write_text(result.model_dump_json(indent=2))
        print(f"\nResult written to {args.json_out}")

    print("\nDecisions:")
    for d in result.decisions:
        icon = "OK" if d.action == "added" else "--"
        print(f"  [{icon}] {d.repo_name}: {d.action}")
        if d.bullet_used:
            print(f"        -> {d.bullet_used}")
        if d.skills_added:
            print(f"        +  skills: {', '.join(d.skills_added)}")


if __name__ == "__main__":
    _cli()