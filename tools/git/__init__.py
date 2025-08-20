"""
Git tools package for PR Review Assistant.

This package provides tools for:
- Detecting changes in Git repositories
- Analyzing diffs and commit history
- Extracting metadata from PRs and commits
"""

from .git_changes import GitChangeDetector

__all__ = ["GitChangeDetector"]
