from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, TypedDict

from loguru import logger

# using loguru's logger
from tools.comparison.code_doc_matcher import CodeDocMatcher
from tools.llm.base import LLMConfig
from tools.llm.tool import create_llm_tool
from tools.schema_analysis.schema_analyzer import (
    SchemaAnalysisInput,
    SchemaAnalysisOutput,
    SchemaAnalyzer,
)

from .base import (
    AgentOutputs,
    DiffInfo,
    OrchestratorState,
    PullRequestInfo,
    RepositoryInfo,
)


class FinalReport(TypedDict):
    """TypedDict describing the final MVP report structure."""

    diff_summary: dict[str, Any]
    code_review: dict[str, Any]
    docs_consistency: dict[str, Any]
    schema_analysis: dict[str, Any]
    summary: str


class MVPOrchestrator:
    """MVP end-to-end orchestrator skeleton.

    The MVP is designed to be swapped out for real agent invocations in a
    future iteration. For now, it simulates the flow and defines the report
    format using the defined interfaces.
    """

    def __init__(self) -> None:
        """Initialize MVP orchestrator instance.

        This constructor currently performs no initialization but is provided to
        document the public API surface for the planner module.
        """
        pass

    async def initialize_workflow(self, pr_url: str) -> OrchestratorState:
        """Workflow initialization: extract PR context from the given URL and
        construct the diff information.

        This method is a placeholder for integrating with Git tooling. In a future
        iteration, it will fetch PR metadata, create the diff descriptor, and initialize
        the OrchestratorState accordingly.

        Args:
            pr_url (str): The URL or identifier of the PR to initialize the workflow for.

        Returns:
            OrchestratorState: The initial orchestrator state containing repo, diff, and PR info.

        Raises:
            Exception: If initialization fails due to invalid input or environment issues.
        """
        repo = RepositoryInfo(provider="unknown", url=pr_url, default_branch="main")
        diff = DiffInfo(
            unified_patch_path=f"{pr_url}.diff",
            changed_files_path=f"{pr_url}.files",
            total_files=None,
            total_additions=None,
            total_deletions=None,
        )
        pr = PullRequestInfo(
            number=0,
            title="MVP 샘플 PR",
            base="main",
            head="feature/mvp",
            author="tester",
            url=pr_url,
        )
        state = OrchestratorState(repo=repo, diff=diff, pr=pr, outputs=AgentOutputs())
        return state

    async def run_mvp_pipeline(self, state: OrchestratorState) -> OrchestratorState:
        """Run the MVP end-to-end pipeline.

        This method orchestrates the sequential/incidental invocation of all MVP
        agents (DiffSummarizer, CodeReviewer, DocsConsistencyChecker, SchemaChangeAnalyst)
        and aggregates their outputs into a final report attached to the provided state.

        Args:
            state (OrchestratorState): The current orchestrator state containing
                PR context, diff metadata, and any existing outputs.

        Returns:
            OrchestratorState: The updated state with the final report and aggregated outputs.

        Raises:
            Exception: If any stage of the pipeline fails and cannot be recovered.
        """
        # Run independent agent invocations in parallel to reduce overall latency
        (
            diff_summary,
            code_review,
            docs_consistency,
            schema_analysis,
        ) = await asyncio.gather(
            self._invoke_diff_summarizer(state),
            self._invoke_code_reviewer(state),
            self._invoke_docs_consistency(state),
            self._invoke_schema_analysis(state),
        )

        final_report: FinalReport = {
            "diff_summary": diff_summary,
            "code_review": code_review,
            "docs_consistency": docs_consistency,
            "schema_analysis": schema_analysis,
            "summary": self._generate_tldr(
                diff_summary, code_review, docs_consistency, schema_analysis
            ),
        }

        # Preserve non-ASCII characters (e.g., Korean) in JSON output
        state.outputs.final_report = json.dumps(
            final_report, indent=2, default=str, ensure_ascii=False
        )
        state.outputs.summary = final_report["summary"]
        state.outputs.code_review = json.dumps(
            code_review, indent=2, default=str, ensure_ascii=False
        )
        state.outputs.docs = json.dumps(
            docs_consistency, indent=2, default=str, ensure_ascii=False
        )
        state.outputs.schema = json.dumps(
            schema_analysis, indent=2, default=str, ensure_ascii=False
        )

        # Attempt to post final report to PR if running in an environment with
        # a configured GitHub token and PR metadata is available.
        try:
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token and state.pr and state.repo:
                # Defer network I/O to a thread to avoid blocking the event loop
                await asyncio.to_thread(self._post_report_to_pr, state, github_token)
        except Exception as e:
            logger.exception("Failed to post final report to PR: %s", e)

        return state

    def _post_report_to_pr(self, state: OrchestratorState, github_token: str) -> None:
        """Synchronous helper to post the final report to the PR using GitHubPoster.

        This method is executed in a background thread to keep the main
        orchestrator event loop responsive.
        """
        try:
            # Lazy import to avoid adding requests dependency to import-time
            from tools.git.provider_github import GitHubPoster, parse_github_pr_url

            # Determine owner/repo/number. Prefer explicit PR URL parsing; fall
            # back to using repo.url + pr.number when available.
            owner = None
            repo = None
            number = None

            if state.pr and state.pr.url:
                try:
                    owner, repo, number = parse_github_pr_url(state.pr.url)
                except ValueError:
                    logger.debug(
                        "Could not parse PR URL, will try repo URL + PR number"
                    )

            if (owner is None or repo is None) and state.repo and state.pr:
                # Attempt to parse owner/repo from repo.url and number from pr.number
                try:
                    owner, repo, _ = parse_github_pr_url(state.repo.url + "/pull/0")
                except Exception:
                    # Last resort: attempt simple split of repo.url
                    url = state.repo.url or ""
                    parts = url.rstrip("/").split("/")
                    if len(parts) >= 2:
                        owner, repo = parts[-2], parts[-1].replace(".git", "")

            if state.pr:
                number = number or getattr(state.pr, "number", None)

            if not (owner and repo and number):
                logger.warning(
                    "Insufficient PR metadata to post report: %s %s",
                    state.repo,
                    state.pr,
                )
                return

            poster = GitHubPoster(token=github_token)
            comment_body = state.outputs.final_report or ""
            # If the report is too large, only post the summary
            if len(comment_body) > 60000:
                comment_body = state.outputs.summary or "Large report - see artifacts"

            resp = poster.post_comment(owner, repo, int(number), comment_body)
            # Record evidence of posting in outputs
            posted = {"posted": True, "response": resp}
            try:
                state.outputs.final_report_post = json.dumps(posted, ensure_ascii=False)
            except Exception:
                state.outputs.final_report_post = str(posted)
        except Exception as e:
            logger.exception("Exception while posting report to PR: %s", e)

    def _generate_tldr(
        self,
        diff_summary: dict[str, Any],
        code_review: dict[str, Any],
        docs_consistency: dict[str, Any],
        schema_analysis: dict[str, Any],
    ) -> str:
        parts = []
        if isinstance(diff_summary, dict):
            tldr = diff_summary.get("tldr")
            if tldr:
                parts.append(tldr)
        return " | ".join(parts)

    async def _invoke_diff_summarizer(self, state: OrchestratorState) -> dict[str, Any]:
        """Invoke the (potential) real DiffSummarizer agent.

        If a DiffSummarizer implementation is available in the repo (as a separate
        agent module), delegate to it. Otherwise, fall back to a sample/mocked response
        to preserve MVP workflow continuity.
        """
        # Attempt to import a real DiffSummarizer if present
        try:
            mod = import_module("agents.change_summarizer")
            if hasattr(mod, "DiffSummarizer"):
                ds = mod.DiffSummarizer()
                # Expected interface: analyze(diff_path, files_path, max_highlights, include_risks)
                if hasattr(ds, "analyze"):
                    result = ds.analyze(
                        diff_path=state.diff.unified_patch_path,
                        files_path=state.diff.changed_files_path,
                        max_highlights=5,
                        include_risks=True,
                    )
                    # If coroutine, await it
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, dict):
                        return result
        except Exception as e:
            logger.exception("DiffSummarizer failed: %s", e)

        # Fallback sample response (MVP placeholder)
        return {
            "tldr": "초안 TL;DR(변경 요약) - MVP",
            "highlights": ["요약 포인트 1", "포인트 2"],
            "risks": [{"description": "샘플 리스크", "level": "medium"}],
            "deployment_impact": "low",
            "compatibility_impact": "none",
            "evidence": [
                {
                    "type": "diff",
                    "target": state.diff.unified_patch_path,
                    "line_range": {"start": 1, "end": 5},
                    "file_path": "docs/mvp_orchestrator_design.md",
                    "description": "샘플 근거",
                    "confidence": 0.9,
                }
            ],
        }

    async def _invoke_code_reviewer(self, state: OrchestratorState) -> dict[str, Any]:
        try:
            # Attempt to import the real CodeReviewer agent
            mod = import_module("agents.reviewer.reviewer")
            if hasattr(mod, "CodeReviewer"):
                # Choose provider based on environment. If no API key is
                # available for a non-mock provider, fall back to mock.
                provider_type = os.getenv("LLM_PROVIDER", "mock")
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
                if provider_type != "mock" and not api_key:
                    logger.warning(
                        "LLM provider '%s' requested but no API key found; using mock instead",
                        provider_type,
                    )
                    provider_type = "mock"

                cfg = LLMConfig(
                    api_key=api_key or "test-key",
                    model=os.getenv("LLM_MODEL", "mock-model"),
                )
                try:
                    llm_tool = create_llm_tool(cfg, provider_type)
                except Exception as e:
                    logger.exception(
                        "Failed to create LLM tool, falling back to mock: %s", e
                    )
                    llm_tool = create_llm_tool(
                        LLMConfig(api_key="test-key", model="mock-model"), "mock"
                    )

                cr = mod.CodeReviewer(llm_tool=llm_tool)
                result = cr.review(
                    diff_path=state.diff.unified_patch_path,
                    files_path=state.diff.changed_files_path,
                    ruleset_version="1.0.0",
                    severity_threshold="low",
                )
                if isinstance(result, dict):
                    return result
        except Exception as e:
            logger.exception("CodeReviewer failed: %s", e)

        # Fallback sample
        return {
            "findings": [],
            "coverage_hints": [],
            "quality_score": 0,
            "security_score": 0,
            "performance_score": 0,
        }

    async def _invoke_docs_consistency(
        self, state: OrchestratorState
    ) -> dict[str, Any]:
        """Invoke the DocsConsistencyChecker if available; otherwise fall back to sample."""
        try:
            matcher = CodeDocMatcher()
            # Use current project path; fallback to repo root
            project_path = "."
            if Path(state.diff.changed_files_path).exists():
                project_path = state.diff.changed_files_path
            # Offload potentially blocking analysis to a thread
            result = await asyncio.to_thread(matcher.analyze_project, project_path)
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.exception("DocsConsistencyChecker failed: %s", e)

        return {
            "mismatches": [],
            "score": 100,
            "missing_docs": [],
            "outdated_docs": [],
            "patch_suggestions": [],
        }

    async def _invoke_schema_analysis(self, state: OrchestratorState) -> dict[str, Any]:
        """Invoke the real SchemaAnalyzer if available; otherwise fall back to sample."""
        try:
            patch_path = state.diff.unified_patch_path
            diff_content = ""
            if patch_path and Path(patch_path).exists():
                # Offload file read to thread to avoid blocking the event loop
                diff_content = await asyncio.to_thread(
                    Path(patch_path).read_text, encoding="utf-8", errors="ignore"
                )
            input_data = SchemaAnalysisInput(
                diff_content=diff_content, schema_files=[], database_type="postgresql"
            )
            analyzer = SchemaAnalyzer()
            # Execute analyzer in a thread if it's synchronous/CPU-bound
            tool_result = await asyncio.to_thread(analyzer.execute, input_data)
            # Normalize to a common 'output' object: prefer the tool_result.output
            # attribute when available; otherwise fall back to None.
            output = getattr(tool_result, "output", None)

            # Defensive extraction with safe fallbacks
            if isinstance(output, SchemaAnalysisOutput):
                ddl_changes_raw = getattr(output, "ddl_changes", None)
                ddl_changes = (
                    [asdict(change) for change in ddl_changes_raw]
                    if ddl_changes_raw
                    else []
                )
                breaking_raw = getattr(output, "breaking_changes", None)
                breaking_changes = (
                    [asdict(b) for b in breaking_raw] if breaking_raw else []
                )
                ops_guide_val = getattr(output, "ops_guide", None)
                ops_guide = asdict(ops_guide_val) if ops_guide_val is not None else None
                mig = getattr(output, "migration_complexity", None)

                # Safely extract enum-like .value when present. Ensure we
                # handle None explicitly to satisfy static type checkers.
                if mig is not None and hasattr(mig, "value"):
                    migration_complexity = mig.value
                elif mig is not None:
                    migration_complexity = str(mig)
                else:
                    migration_complexity = "low"
                total_impact_score = getattr(output, "total_impact_score", 0.0)
            else:
                ddl_changes = []
                breaking_changes = []
                ops_guide = None
                migration_complexity = "low"
                total_impact_score = 100.0

            return {
                "ddl_changes": ddl_changes,
                "breaking_changes": breaking_changes,
                "ops_guide": ops_guide,
                "migration_complexity": migration_complexity,
                "total_impact_score": total_impact_score,
            }
        except Exception as e:
            logger.exception("Schema analysis failed: %s", e)
            return {
                "ddl_changes": [],
                "breaking_changes": [],
                "ops_guide": None,
                "migration_complexity": "low",
                "total_impact_score": 100.0,
            }
