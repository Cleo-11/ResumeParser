"""
ingestion/github_ingestor.py
Layer 1 - Ingestion: Pull all repo data from GitHub APIs.
"""

from __future__ import annotations

import base64
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

import requests

from resume_pipeline.core.logger import LayerLogger
from resume_pipeline.core.models import RawRepo

GITHUB_API = "https://api.github.com"
_log = LayerLogger("ingestion")
_MAX_RETRIES = 3


def ingest(
    username: str,
    token: Optional[str] = None,
    days_back: int = 90,
    max_repos: int = 10,
    include_forks: bool = False,
) -> List[RawRepo]:
    _log.info(f"Starting ingestion for @{username}", days_back=days_back, max_repos=max_repos)
    headers = _make_headers(token)

    repos_json = _fetch_repos(username, headers)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    recent = [
        r for r in repos_json
        if _parse_dt(r["pushed_at"]) > cutoff
        and (include_forks or not r.get("fork", False))
    ][:max_repos]

    _log.info(f"Found {len(recent)} repos within window (from {len(repos_json)} total)")

    raw_repos: List[RawRepo] = []
    for r in recent:
        _log.debug(f"Ingesting {r['name']} ...")
        raw = _build_raw_repo(r, headers, username)
        raw_repos.append(raw)
        time.sleep(0.3)

    _log.success(f"Ingestion complete - {len(raw_repos)} repos")
    return raw_repos


def _fetch_repos(username: str, headers: dict) -> list:
    resp = _get(f"{GITHUB_API}/users/{username}/repos", headers,
                params={"sort": "pushed", "direction": "desc", "per_page": 100})
    return resp or []


def _fetch_readme(owner: str, repo: str, headers: dict) -> str:
    data = _get(f"{GITHUB_API}/repos/{owner}/{repo}/readme", headers)
    if not data:
        return ""
    try:
        content = data.get("content", "")
        return base64.b64decode(content.replace("\n", "")).decode("utf-8", errors="ignore")[:4000]
    except Exception:
        return ""


def _fetch_commits(owner: str, repo: str, headers: dict, n: int = 20) -> List[str]:
    data = _get(f"{GITHUB_API}/repos/{owner}/{repo}/commits", headers,
                params={"per_page": n}) or []
    return [c["commit"]["message"].split("\n")[0] for c in data if c.get("commit")]


def _fetch_languages(owner: str, repo: str, headers: dict) -> Dict[str, int]:
    return _get(f"{GITHUB_API}/repos/{owner}/{repo}/languages", headers) or {}


def _fetch_contributors_count(owner: str, repo: str, headers: dict) -> int:
    """Returns contributor count by fetching up to 100 at once (covers most personal projects)."""
    data = _get(
        f"{GITHUB_API}/repos/{owner}/{repo}/contributors",
        headers,
        params={"per_page": 100, "anon": "true"},
    ) or []
    return len(data)


def _check_has_tests(owner: str, repo: str, headers: dict) -> bool:
    data = _get(f"{GITHUB_API}/repos/{owner}/{repo}/contents", headers) or []
    test_names = {"test", "tests", "spec", "__tests__", "testing"}
    return any(item.get("name", "").lower() in test_names for item in data)


def _check_has_ci(owner: str, repo: str, headers: dict) -> bool:
    data = _get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/.github", headers)
    return data is not None


def _build_raw_repo(r: dict, headers: dict, username: str) -> RawRepo:
    owner = r["owner"]["login"]
    name  = r["name"]

    readme    = _fetch_readme(owner, name, headers)
    commits   = _fetch_commits(owner, name, headers)
    langs     = _fetch_languages(owner, name, headers)
    contrib   = _fetch_contributors_count(owner, name, headers)
    has_tests = _check_has_tests(owner, name, headers)
    has_ci    = _check_has_ci(owner, name, headers)
    license_name = (r.get("license") or {}).get("name")

    return RawRepo(
        name=name,
        full_name=r["full_name"],
        description=r.get("description"),
        url=r["html_url"],
        language=r.get("language"),
        topics=r.get("topics") or [],
        stars=r.get("stargazers_count", 0),
        forks=r.get("forks_count", 0),
        open_issues=r.get("open_issues_count", 0),
        pushed_at=_parse_dt(r["pushed_at"]),
        created_at=_parse_dt(r["created_at"]),
        readme=readme,
        recent_commits=commits,
        languages_breakdown=langs,
        contributors_count=contrib,
        has_tests=has_tests,
        has_ci=has_ci,
        license_name=license_name,
    )


def _make_headers(token: Optional[str]) -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(url: str, headers: dict, params: Optional[dict] = None):
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < _MAX_RETRIES - 1:
                    wait = int(resp.headers.get("Retry-After", (2 ** attempt) + random.uniform(0, 1)))
                    _log.warn(f"GitHub rate limit / server error ({resp.status_code}) — retrying in {wait}s")
                    time.sleep(wait)
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < _MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                _log.warn(f"GET {url} failed after {_MAX_RETRIES} attempts: {e}")
                return None
    return None


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
