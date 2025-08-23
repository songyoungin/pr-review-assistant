"""
Schema Analysis Tools Package

This package provides comprehensive tools for analyzing database schema changes,
detecting breaking changes, and generating operational guides for deployments.
"""

from .schema_analyzer import (
    BreakingChange,
    DDLChange,
    OpsGuide,
    SchemaAnalysisInput,
    SchemaAnalysisOutput,
    SchemaAnalyzer,
)

__all__ = [
    "SchemaAnalyzer",
    "DDLChange",
    "BreakingChange",
    "OpsGuide",
    "SchemaAnalysisInput",
    "SchemaAnalysisOutput",
]

__version__ = "0.1.0"
