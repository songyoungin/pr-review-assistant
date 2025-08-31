"""Package-level exports for the change_summarizer agent.

The actual implementation of `DiffSummarizer` lives in
`agents.change_summarizer.summarizer` to keep package initialization
lightweight and testable. This module re-exports the implementation
so `import_module("agents.change_summarizer")` and attribute lookup
used by the planner continues to work without change.
"""

from __future__ import annotations

from .summarizer import DiffSummarizer

__all__ = ["DiffSummarizer"]
