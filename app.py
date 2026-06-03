"""
app.py
Streamlit web interface for the Resume Pipeline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import Pipeline, PipelineConfig

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Pipeline",
    page_icon="📄",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Credentials")
    github_username = st.text_input(
        "GitHub Username *",
        placeholder="octocat",
    )
    openai_key = st.text_input(
        "OpenAI API Key *",
        type="password",
        placeholder="sk-proj-...",
    )
    github_token = st.text_input(
        "GitHub Token (optional)",
        type="password",
        help=(
            "Personal access token — raises the GitHub rate limit from "
            "60 req/hr (unauthenticated) to 5,000 req/hr. "
            "Needed if you have more than ~8 repos to scan."
        ),
    )

    st.divider()
    st.subheader("Pipeline Options")

    model = st.selectbox(
        "OpenAI Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="gpt-4o-mini is ~15× cheaper and fast enough for most resumes.",
    )
    days_back = st.slider("Days to look back", min_value=30, max_value=365, value=90)
    max_repos = st.slider("Max repos to process", min_value=1, max_value=15, value=8)
    min_confidence = st.slider(
        "Min bullet confidence",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Bullets scored below this threshold are skipped.",
    )
    include_forks = st.checkbox("Include forked repos", value=False)
    dry_run = st.checkbox(
        "Dry run (preview only)",
        value=False,
        help="Runs the full pipeline but does not write the output file.",
    )

    st.divider()
    st.caption(
        "Each repo costs ~3 OpenAI API calls (understand → transform → style-check). "
        "For 8 repos expect roughly 1–3 minutes and $0.10–$0.50 depending on the model."
    )

# ── Main area ──────────────────────────────────────────────────────────────
st.title("📄 Resume Pipeline")
st.markdown(
    "Upload your `.docx` resume. The pipeline scans your recent GitHub repos, "
    "generates ATS-friendly bullet points using GPT, and merges them into your resume — "
    "with duplicate detection and style-matching."
)

uploaded_file = st.file_uploader(
    "Upload your resume (.docx)",
    type=["docx"],
    help="Only Word .docx files are supported.",
)

st.divider()
run_clicked = st.button("▶  Update my resume", type="primary", use_container_width=False)

if run_clicked:
    # ── Validation ─────────────────────────────────────────────────────────
    missing = []
    if not github_username:
        missing.append("GitHub username")
    if not openai_key:
        missing.append("OpenAI API key")
    if not uploaded_file:
        missing.append("resume file")
    if missing:
        st.error(f"Please provide: {', '.join(missing)}.")
        st.stop()

    # ── Temp file setup ────────────────────────────────────────────────────
    tmp_in = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    tmp_in.write(uploaded_file.getvalue())
    tmp_in.close()
    tmp_out_path = tmp_in.name[:-5] + "_updated.docx"

    result = None
    result_bytes: bytes | None = None

    try:
        # ── Pipeline execution with live log ───────────────────────────────
        with st.status("Running pipeline…", expanded=True) as status_box:
            log_placeholder = st.empty()
            log_lines: list[str] = []

            LEVEL_ICON = {
                "SUCCESS": "✅",
                "WARN":    "⚠️",
                "ERROR":   "❌",
                "DEBUG":   "🔍",
                "INFO":    "→",
            }

            def on_event(event: dict) -> None:
                level = event.get("level", "INFO")
                msg   = event.get("msg", "")
                icon  = LEVEL_ICON.get(level, "→")
                log_lines.append(f"{icon} {msg}")
                log_placeholder.markdown(
                    "\n\n".join(log_lines[-35:]),
                    unsafe_allow_html=False,
                )

            config = PipelineConfig(
                github_username=github_username,
                openai_api_key=openai_key,
                github_token=github_token or None,
                resume_path=tmp_in.name,
                output_path=tmp_out_path,
                model=model,
                days_back=days_back,
                max_repos=max_repos,
                min_confidence=min_confidence,
                include_forks=include_forks,
                dry_run=dry_run,
                event_handlers=[on_event],
            )

            result = Pipeline(config).run()
            status_box.update(label="Pipeline complete ✅", state="complete", expanded=False)

        # Read output file bytes before cleanup
        if Path(tmp_out_path).exists():
            result_bytes = Path(tmp_out_path).read_bytes()

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")

    finally:
        for path in [tmp_in.name, tmp_out_path]:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ── Results display ────────────────────────────────────────────────────
    if result:
        st.subheader("Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Repos scanned",  result.repos_ingested)
        c2.metric("Bullets added",  result.total_bullets_added)
        c3.metric("Skills added",   result.total_skills_added)
        c4.metric("Duration",       f"{result.duration_seconds:.0f}s")

        if result.warnings:
            for w in result.warnings:
                st.warning(w)

        if result.decisions:
            st.subheader("What changed")
            added   = [d for d in result.decisions if d.action == "added"]
            skipped = [d for d in result.decisions if d.action != "added"]

            if added:
                st.markdown("**Added to resume:**")
                for d in added:
                    with st.expander(f"✅ {d.repo_name}", expanded=False):
                        st.markdown(f"**Bullet:** {d.bullet_used}")
                        if d.skills_added:
                            st.markdown(f"**Skills added:** {', '.join(d.skills_added)}")

            if skipped:
                st.markdown("**Skipped:**")
                for d in skipped:
                    icon = "⏭" if d.action == "skipped_duplicate" else "⚠"
                    st.info(f"{icon} **{d.repo_name}** — {d.action.replace('_', ' ')}: {d.reason}")

        st.divider()

        if result_bytes and not dry_run:
            base_name = uploaded_file.name.removesuffix(".docx")
            st.download_button(
                label="⬇  Download Updated Resume",
                data=result_bytes,
                file_name=f"{base_name}_updated.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=False,
            )
        elif dry_run:
            st.info(
                "Dry run complete — no file was written. "
                "Uncheck **Dry run** in the sidebar and click Run again to get the updated DOCX."
            )
