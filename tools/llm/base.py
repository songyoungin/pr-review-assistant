"""
LLM provider base interface and data structures.

This module defines the base interface and common data structures that all
LLM providers must implement.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LLMConfig:
    """
    LLM provider configuration structure.

    Contains API key, model settings, parameters for each provider.
    """

    api_key: str  # API key
    model: str  # Model name
    base_url: str | None = None  # Custom API endpoint
    timeout: int = 60  # Request timeout (seconds)
    max_tokens: int = 4096  # Maximum token count
    temperature: float = 0.0  # Generation temperature (deterministic)
    top_p: float = 1.0  # Nucleus sampling parameter
    frequency_penalty: float = 0.0  # Frequency penalty
    presence_penalty: float = 0.0  # Presence penalty

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class LLMMessage:
    """
    LLM conversation message structure.
    """

    role: str  # 'system', 'user', 'assistant'
    content: str  # Message content
    metadata: dict[str, Any] | None = None  # Additional metadata

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        result = {"role": self.role, "content": self.content}
        if self.metadata:
            result["metadata"] = str(self.metadata)
        return result


@dataclass
class LLMUsage:
    """
    LLM API usage information.
    """

    prompt_tokens: int  # Number of prompt tokens
    completion_tokens: int  # Number of completion tokens
    total_tokens: int  # Total token count
    estimated_cost: float | None = None  # Estimated cost

    def to_dict(self) -> dict[str, Any]:
        """Convert usage info to dictionary."""
        return asdict(self)


@dataclass
class LLMResponse:
    """
    LLM API response structure.
    """

    content: str  # Generated content
    usage: LLMUsage  # Token usage
    model: str  # Model used
    finish_reason: str  # Generation completion reason
    citations: list[dict[str, Any]] | None = None  # Citation information
    function_calls: list[dict[str, Any]] | None = None  # Function calls
    metadata: dict[str, Any] | None = None  # Additional metadata

    def __post_init__(self) -> None:
        if self.citations is None:
            self.citations = []
        if self.function_calls is None:
            self.function_calls = []

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        result = asdict(self)
        result["usage"] = self.usage.to_dict()
        return result

    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class LLMProvider(ABC):
    """
    Base interface for LLM providers.

    All LLM providers must implement this interface.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = self.__class__.__name__

    @abstractmethod
    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """
        Generate text using LLM.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate provider connection.

        Returns:
            True if connection is valid
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to analyze

        Returns:
            Estimated token count
        """
        # Simple estimation: 1 token ≈ 4 characters for English
        return len(text) // 4

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider information."""
        return {
            "provider_name": self.provider_name,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
