"""
LLM integration tools - NFR-2, NFR-5 compliance.

This package integrates various LLM providers and performs rule-based code analysis and review.
All results include evidence (file:line) and rule ID/version for traceability.
"""

from .base import LLMConfig, LLMProvider, LLMResponse
from .prompts import PromptManager, PromptTemplate
from .providers import AnthropicProvider, OpenAIProvider
from .tool import LLMTool

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "LLMTool",
    "OpenAIProvider",
    "AnthropicProvider",
    "PromptTemplate",
    "PromptManager",
]
