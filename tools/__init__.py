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
from .rules import (
    RuleCategory,
    RuleDefinition,
    RuleRegistry,
    Severity,
    get_rule_info,
    get_rule_registry,
    validate_evidence_rule_compliance,
)
from .schema_analysis.schema_analyzer import SchemaAnalyzer

__all__ = [
    # Base classes and types
    "BaseTool",
    "ToolResult",
    "ToolEvidence",
    "ToolMetrics",
    "ToolStatus",
    "ToolErrorCode",
    # Rule management
    "RuleRegistry",
    "RuleDefinition",
    "RuleCategory",
    "Severity",
    "get_rule_registry",
    "validate_evidence_rule_compliance",
    "get_rule_info",
    # Concrete tools
    "GitChangeDetector",
    "PythonASTAnalyzer",
    "CodeDocMatcher",
    "SchemaAnalyzer",
]
