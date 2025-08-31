"""Tests for the CodeReviewer agent using the mock LLM provider."""

from __future__ import annotations

from pathlib import Path

from agents.reviewer.reviewer import CodeReviewer
from tools.llm.base import LLMConfig
from tools.llm.providers import MockProvider
from tools.llm.tool import LLMTool


def test_reviewer_success(tmp_path: Path) -> None:
    """CodeReviewer returns parsed LLMAnalysisResult when LLM returns valid JSON."""

    # Prepare a sample diff file
    diff_text = "--- a/test.py\n+++ b/test.py\n@@ -0,0 +1 @@\n+def test_function():\n"

    p = tmp_path / "sample.diff"
    p.write_text(diff_text, encoding="utf-8")

    # Create a mock JSON response wrapped in ```json block that LLMTool will parse
    json_response = (
        "```json\n"
        "{\n"
        '  "findings": [\n'
        "    {\n"
        '      "file_path": "test.py",\n'
        '      "line_number": 1,\n'
        '      "rule_id": "QUAL001",\n'
        '      "severity": "low",\n'
        '      "message": "Test finding"\n'
        "    }\n"
        "  ],\n"
        '  "summary": "Test analysis completed"\n'
        "}\n"
        "```"
    )

    cfg = LLMConfig(api_key="test", model="mock-model")
    provider = MockProvider(cfg, mock_responses=[json_response])
    llm_tool = LLMTool(provider=provider)

    reviewer = CodeReviewer(llm_tool=llm_tool)
    result = reviewer.review(str(p), None)

    assert isinstance(result, dict)
    # LLMAnalysisResult.to_dict() fields expected
    assert result.get("analysis_type") == "code_review"
    findings = result.get("findings")
    assert isinstance(findings, list)
    assert len(findings) == 1
    f = findings[0]
    assert f.get("file_path") == "test.py"
    assert f.get("rule_id") == "QUAL001"


def test_reviewer_invalid_response(tmp_path: Path) -> None:
    """When LLM returns invalid content reviewer should return default empty structure."""

    diff_text = "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+x = 1\n"

    p = tmp_path / "x.diff"
    p.write_text(diff_text, encoding="utf-8")

    cfg = LLMConfig(api_key="test", model="mock-model")
    provider = MockProvider(cfg, mock_responses=["Not JSON"])
    llm_tool = LLMTool(provider=provider)

    reviewer = CodeReviewer(llm_tool=llm_tool)
    result = reviewer.review(str(p), None)

    # Should return normalized empty review structure
    assert isinstance(result, dict)
    assert result.get("findings") == []
    assert result.get("quality_score") == 0
