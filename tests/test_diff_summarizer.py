"""Unit tests for the DiffSummarizer agent.

These tests validate basic parsing, highlight selection, risk detection
and evidence generation for small unified diff inputs.
"""

from __future__ import annotations

from pathlib import Path

from agents.change_summarizer import DiffSummarizer


def test_analyze_empty_diff() -> None:
    """When no diff is provided the summarizer returns a conservative summary."""

    ds = DiffSummarizer()
    result = ds.analyze(None, None)

    assert isinstance(result, dict)
    assert "tldr" in result
    assert "No significant additions" in result["tldr"]
    assert result["highlights"] == []
    assert result["risks"] == []
    assert result["deployment_impact"] == "low"


def test_parse_simple_diff(tmp_path: Path) -> None:
    """Parse a small unified diff with two files and verify highlights/evidence."""

    diff_text = (
        "diff --git a/foo.py b/foo.py\n"
        "index 000..111\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+def foo():\n"
        "+    return 1\n"
        "+\n"
        "diff --git a/bar.txt b/bar.txt\n"
        "--- a/bar.txt\n"
        "+++ b/bar.txt\n"
        "@@ -0,0 +1,1 @@\n"
        "+hello\n"
    )

    p = tmp_path / "sample.diff"
    p.write_text(diff_text, encoding="utf-8")

    ds = DiffSummarizer()
    result = ds.analyze(str(p), None, max_highlights=2)

    assert "highlights" in result
    highlights = result["highlights"]
    assert len(highlights) >= 1
    # top highlight should mention foo.py with 3 additions
    assert "foo.py" in highlights[0]
    assert "3 additions" in highlights[0]

    # evidence should reference foo.py and bar.txt
    evidence_files = {e.get("file_path") for e in result["evidence"]}
    assert "foo.py" in evidence_files
    assert "bar.txt" in evidence_files


def test_detect_risk_drop_table(tmp_path: Path) -> None:
    """Detect 'DROP TABLE' as a high severity risk and set impact accordingly."""

    diff_text = (
        "--- a/migrations.sql\n"
        "+++ b/migrations.sql\n"
        "@@ -0,0 +1,1 @@\n"
        "+DROP TABLE users;\n"
    )

    p = tmp_path / "migrate.diff"
    p.write_text(diff_text, encoding="utf-8")

    ds = DiffSummarizer()
    result = ds.analyze(str(p), None)

    # There should be at least one detected risk and it should be high
    risks = result.get("risks", [])
    assert len(risks) >= 1
    assert any(r.get("level") == "high" for r in risks)

    assert result.get("deployment_impact") == "high"
    assert result.get("compatibility_impact") == "breaking"
