"""
Comparison tools package for PR Review Assistant.

This package provides tools for:
- Comparing code changes with documentation
- Detecting mismatches between implementation and docs
- Suggesting documentation updates
"""

from .code_doc_matcher import CodeDocMatcher

__all__ = ["CodeDocMatcher"]
