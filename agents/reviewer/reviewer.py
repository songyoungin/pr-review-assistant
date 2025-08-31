"""Code reviewer agent that wraps the LLM analysis tool.

This module implements `CodeReviewer`, a light wrapper which prepares
inputs (diff and files), calls the `LLMTool` for `code_review` analysis
and normalizes the output into a planner-friendly dictionary.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, cast

from tools.base import ToolResult, ToolStatus
from tools.llm.base import LLMConfig
from tools.llm.tool import LLMAnalysisRequest, LLMTool, create_llm_tool


class CodeReviewer:
    """Wrapper for LLM-based code review analysis.

    The class is intentionally lightweight: it constructs an
    `LLMAnalysisRequest`, calls the provided `LLMTool` and returns a
    plain dictionary compatible with the planner's expectations.
    """

    def __init__(
        self, llm_tool: LLMTool | None = None, provider_type: str = "mock"
    ) -> None:
        """Initialize CodeReviewer.

        Args:
            llm_tool: Optional preconfigured LLMTool instance. If not
                provided a mock provider will be created via
                `create_llm_tool` for local testing.
            provider_type: Provider type string passed to `create_llm_tool`
                when a new tool is created. Defaults to "mock".
        """

        if llm_tool is not None:
            self.llm_tool = llm_tool
        else:
            cfg = LLMConfig(api_key="test-key", model="mock-model")
            self.llm_tool = create_llm_tool(cfg, provider_type)

        # If the provider's generate is an async function, provide a
        # synchronous wrapper to allow synchronous execution in tests.
        try:
            provider = getattr(self.llm_tool, "provider", None)
            if provider and inspect.iscoroutinefunction(
                getattr(provider, "generate", None)
            ):
                orig = provider.generate

                def _sync_generate(messages: list[Any], **kwargs: Any) -> Any:
                    return asyncio.run(orig(messages, **kwargs))

                # Monkeypatch the provider's generate with the sync wrapper
                provider.generate = _sync_generate
        except Exception:
            # Best-effort; do not fail initialization if monkeypatching is not possible
            pass

    def review(
        self,
        diff_path: str | None,
        files_path: str | None,
        ruleset_version: str = "1.0.0",
        severity_threshold: str = "low",
    ) -> dict[str, Any]:
        """Run a code review analysis and return normalized results.

        Args:
            diff_path: Path to a unified diff file (may be None).
            files_path: (Unused for MVP) Path that may point to changed files.
            ruleset_version: Ruleset version to include in LLM context.
            severity_threshold: Minimum severity threshold for findings.

        Returns:
            A dictionary containing analysis results. On success the
            dictionary will be the serialized `LLMAnalysisResult` from
            `LLMTool`. On failure a default empty structure is returned.
        """

        diff_text = ""
        if diff_path:
            try:
                p = Path(diff_path)
                if p.exists():
                    diff_text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                diff_text = ""

        # Minimal files payload: include diff as single pseudo-file so
        # LLM has access to content. Also extract changed file paths from
        # the unified diff and include them with empty content so that
        # LLMAnalysisResult findings referencing those paths are accepted
        # by the validation logic.
        files_payload: list[dict[str, Any]] = [
            {"path": diff_path or "<inline>", "content": diff_text, "language": "text"}
        ]

        # Extract changed file paths from diff headers (lines like '+++ b/path')
        changed_files_from_diff: list[str] = []
        for line in diff_text.splitlines():
            if line.startswith("+++"):
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[1]
                    token = token.replace("b/", "")
                    # Use basename for LLM compatibility (e.g. 'test.py')
                    changed_files_from_diff.append(Path(token).name)

        # Add unique changed files to payload with empty content if not present
        for cf in dict.fromkeys(changed_files_from_diff):
            files_payload.append({"path": cf, "content": "", "language": "text"})

        # Prepare context variables expected by prompt templates
        changed_files_list: list[str] = [
            str(fp.get("path")) for fp in files_payload if fp.get("path")
        ]
        request = LLMAnalysisRequest(
            analysis_type="code_review",
            content=diff_text,
            context={
                "changed_files": "\n".join(changed_files_list),
                "diff_content": diff_text,
                "ruleset_version": ruleset_version,
                "severity_threshold": severity_threshold,
            },
            files=files_payload,
            metadata={"source": "CodeReviewer"},
        )

        try:
            tool_result: ToolResult = self.llm_tool.execute(request)
            if tool_result.status == ToolStatus.SUCCESS and tool_result.output:
                # Normalize output and ensure analysis_type is present
                try:
                    out = tool_result.output
                    out_dict = cast(dict[str, Any], out.to_dict())
                    # Force analysis_type to the request value if missing
                    if not out_dict.get("analysis_type"):
                        out_dict["analysis_type"] = request.analysis_type
                    return out_dict
                except Exception:
                    # Fallback to raw dataclass conversion
                    return getattr(tool_result.output, "to_dict", lambda: {})()
            # On error or partial results, return a normalized empty shape
            return {
                "findings": [],
                "coverage_hints": [],
                "quality_score": 0,
                "security_score": 0,
                "performance_score": 0,
            }

        except Exception:
            # Fail-safe: return an empty code review structure
            return {
                "findings": [],
                "coverage_hints": [],
                "quality_score": 0,
                "security_score": 0,
                "performance_score": 0,
            }


__all__ = ["CodeReviewer"]
