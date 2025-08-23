"""
Tools package for PR Review Assistant.

This package contains various tools for:
- Git operations and change detection
- Code analysis and AST parsing
- Code-documentation matching and validation
"""

from .base import (
    BaseTool,
    ToolErrorCode,
    ToolEvidence,
    ToolMetrics,
    ToolResult,
    ToolStatus,
)
from .code_analysis.python_ast import PythonASTAnalyzer
from .comparison.code_doc_matcher import CodeDocMatcher
from .git.git_changes import GitChangeDetector
from .schema_analysis.schema_analyzer import SchemaAnalyzer

__all__ = [
    # Base classes and types
    "BaseTool",
    "ToolResult",
    "ToolEvidence",
    "ToolMetrics",
    "ToolStatus",
    "ToolErrorCode",
    # Concrete tools
    "GitChangeDetector",
    "PythonASTAnalyzer",
    "CodeDocMatcher",
    "SchemaAnalyzer",
]
