"""Implementation module for DiffSummarizer.

This file contains the full implementation extracted from the package
`__init__` to improve testability and maintainability.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class DiffSummarizer:
    """Simple diff summarizer used by the MVP orchestrator.

    The class is intentionally small: it reads a unified diff file (if
    present), extracts added lines per-file, produces a short TL;DR,
    highlights the files with the most additions, detects a small set
    of high/medium/low risk patterns and returns evidence items.
    """

    def analyze(
        self,
        diff_path: str | None,
        files_path: str | None,
        max_highlights: int = 5,
        include_risks: bool = True,
    ) -> dict[str, Any]:
        """Analyze a unified diff and produce a summary.

        Args:
            diff_path: Path to a unified diff file. If the file does not
                exist or is empty the analyzer will return a conservative
                fallback summary.
            files_path: (Unused in MVP) Path listing changed files. Kept
                for API compatibility with the planner.
            max_highlights: Maximum number of highlighted files to
                include in the `highlights` list.
            include_risks: If True, run a small rule-set to detect
                potential risks (schema drops, sensitive tokens, exec).

        Returns:
            A dictionary containing the following keys expected by the
            orchestrator: `tldr`, `highlights`, `risks`,
            `deployment_impact`, `compatibility_impact`, `evidence`.
        """

        diff_text = ""
        if diff_path:
            try:
                p = Path(diff_path)
                if p.exists():
                    diff_text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # Swallow errors and fall back to empty diff_text
                diff_text = ""

        # Parse diff and collect added lines per file
        file_summaries = self._parse_diff(diff_text)

        # Build highlights: top files by additions
        sorted_files = sorted(
            file_summaries.items(), key=lambda kv: kv[1][0], reverse=True
        )

        highlights: list[str] = []
        for file_path, (added_count, snippets, _first_idx) in sorted_files[
            :max_highlights
        ]:
            snippet_preview = snippets[0] if snippets else ""
            highlights.append(
                f"{file_path} ({added_count} additions): {snippet_preview}"
            )

        # Detect risks
        risks: list[dict[str, Any]] = []
        if include_risks:
            risks = self._detect_risks(file_summaries)

        deployment_impact, compatibility_impact = self._assess_impact(risks)

        evidence = self._build_evidence(file_summaries, risks, diff_path)

        # TL;DR: combine top highlight and top risk
        tldr_parts: list[str] = []
        if highlights:
            tldr_parts.append(f"Top changes: {highlights[0]}")
        if risks:
            top_risk = risks[0]
            tldr_parts.append(f"Top risk: {top_risk.get('description')}")
        if not tldr_parts:
            tldr_parts.append("No significant additions detected in diff.")

        return {
            "tldr": " | ".join(tldr_parts),
            "highlights": highlights,
            "risks": risks,
            "deployment_impact": deployment_impact,
            "compatibility_impact": compatibility_impact,
            "evidence": evidence,
        }

    def _parse_diff(self, diff_text: str) -> dict[str, tuple[int, list[str], int]]:
        """Parse unified diff text and return per-file summaries.

        The returned mapping is: file_path -> (added_count, snippets, first_added_index)
        where `snippets` is a short list of added-line strings (trimmed).
        """

        summaries: dict[str, tuple[int, list[str], int]] = {}
        current_file: str | None = None
        # Track the first addition index (approximate) per file
        first_index_map: dict[str, int] = {}
        line_index = 0

        for raw in diff_text.splitlines():
            line_index += 1
            # File header lines look like: '+++ b/path/to/file' or '+++ path/to/file'
            if raw.startswith("+++"):
                parts = raw.split()
                if len(parts) >= 2:
                    path_token = parts[1]
                    # strip leading a/ or b/ if present
                    current_file = re.sub(r"^([ab]/)", "", path_token)
                    # initialize
                    if current_file not in summaries:
                        summaries[current_file] = (0, [], 0)
                        first_index_map[current_file] = 0
                continue

            # Added lines start with a single '+' but not '+++'
            if raw.startswith("+") and not raw.startswith("+++"):
                if not current_file:
                    # Unknown file context; attribute to <unknown>
                    current_file = "<unknown>"
                    if current_file not in summaries:
                        summaries[current_file] = (0, [], 0)
                        first_index_map[current_file] = 0

                added_line = raw[1:].strip()
                added_count, snippets, first_idx = summaries[current_file]
                # store small snippets (limit 5)
                if len(snippets) < 5 and added_line:
                    snippets.append(added_line[:200])
                added_count += 1
                if first_index_map.get(current_file, 0) == 0:
                    first_index_map[current_file] = line_index

                summaries[current_file] = (
                    added_count,
                    snippets,
                    first_index_map[current_file],
                )

        return summaries

    def _detect_risks(
        self, summaries: dict[str, tuple[int, list[str], int]]
    ) -> list[dict[str, Any]]:
        """Detect simple rule-based risks in added snippets.

        Returns a list of risk dictionaries sorted by severity (high -> low).
        """

        risk_patterns = [
            (
                re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
                "high",
                "Table removal detected",
            ),
            (
                re.compile(r"\bDROP\s+COLUMN\b", re.IGNORECASE),
                "high",
                "Column removal detected",
            ),
            (
                re.compile(r"ALTER\s+COLUMN.*TYPE", re.IGNORECASE),
                "high",
                "Column type change detected",
            ),
            (
                re.compile(r"\bpassword\b|\bsecret\b|\btoken\b", re.IGNORECASE),
                "high",
                "Potential sensitive data in code",
            ),
            (
                re.compile(r"\bexec\(|\bsystem\(|\bsubprocess\.", re.IGNORECASE),
                "medium",
                "Possible shell/exec usage",
            ),
        ]

        risks: list[dict[str, Any]] = []

        for file_path, (added_count, snippets, first_idx) in summaries.items():
            for snippet in snippets:
                for pattern, level, desc in risk_patterns:
                    if pattern.search(snippet):
                        confidence = 0.9 if level == "high" else 0.7
                        risks.append(
                            {
                                "description": f"{desc}: '{snippet[:120]}'",
                                "level": level,
                                "file_path": file_path,
                                "line_range": {
                                    "start": first_idx or 1,
                                    "end": (first_idx or 1) + added_count - 1,
                                },
                                "confidence": confidence,
                            }
                        )

        # sort risks: high first
        risks.sort(
            key=lambda r: {"high": 0, "medium": 1, "low": 2}.get(r.get("level", ""), 3)
        )
        return risks

    def _assess_impact(self, risks: list[dict[str, Any]]) -> tuple[str, str]:
        """Assess deployment and compatibility impact from detected risks.

        Returns a tuple (deployment_impact, compatibility_impact).
        """

        if any(r.get("level") == "high" for r in risks):
            return ("high", "breaking")
        if any(r.get("level") == "medium" for r in risks):
            return ("medium", "backward")
        return ("low", "none")

    def _build_evidence(
        self,
        summaries: dict[str, tuple[int, list[str], int]],
        risks: list[dict[str, Any]],
        diff_path: str | None,
    ) -> list[dict[str, Any]]:
        """Create evidence items for highlights and risks.

        Evidence items follow the shape planner expects (type, target, line_range, file_path, description, confidence).
        """

        evidence: list[dict[str, Any]] = []

        for file_path, (added_count, _snippets, first_idx) in summaries.items():
            if added_count <= 0:
                continue
            evidence.append(
                {
                    "type": "diff",
                    "target": diff_path or "<inline>",
                    "line_range": {
                        "start": first_idx or 1,
                        "end": (first_idx or 1) + added_count - 1,
                    },
                    "file_path": file_path,
                    "description": f"{added_count} added lines in {file_path}",
                    "confidence": 0.8,
                }
            )

        # Also add one evidence item per detected risk (if not duplicate)
        for r in risks:
            evidence.append(
                {
                    "type": "diff",
                    "target": diff_path or "<inline>",
                    "line_range": r.get("line_range", {"start": 1, "end": 1}),
                    "file_path": r.get("file_path", "<unknown>"),
                    "description": r.get("description"),
                    "confidence": float(r.get("confidence", 0.6)),
                }
            )

        return evidence


__all__ = ["DiffSummarizer"]
