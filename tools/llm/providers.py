"""
LLM provider implementations.

Contains specific implementations for various LLM providers like OpenAI, Anthropic.
"""

from typing import Any

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import LLMConfig, LLMMessage, LLMProvider, LLMResponse, LLMUsage


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider implementation.

    Supports GPT-4, GPT-3.5-turbo and other OpenAI models.
    """

    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=3,
        )

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate text using OpenAI API."""

        # Convert messages
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Set request parameters
        request_params = {
            "model": self.config.model,
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.config.frequency_penalty
            ),
            "presence_penalty": kwargs.get(
                "presence_penalty", self.config.presence_penalty
            ),
        }

        try:
            # API call
            response = await self.client.chat.completions.create(**request_params)

            # Convert response
            choice = response.choices[0]
            usage = response.usage

            llm_usage = LLMUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost=self._calculate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                ),
            )

            return LLMResponse(
                content=choice.message.content or "",
                usage=llm_usage,
                model=response.model,
                finish_reason=choice.finish_reason,
                metadata={"provider": "openai"},
            )

        except Exception as e:
            # Error handling and retry logic
            raise RuntimeError(f"OpenAI API call failed: {str(e)}") from e

    async def validate_connection(self) -> bool:
        """Validate OpenAI API connection."""
        try:
            # Simple test request
            test_messages = [LLMMessage(role="user", content="Hello")]
            await self.generate(test_messages, max_tokens=10)
            return True
        except Exception:
            return False

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate OpenAI API cost."""
        # Pricing by model (USD per 1K tokens)
        pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
        }

        model_pricing = pricing.get(
            self.config.model, {"prompt": 0.002, "completion": 0.002}
        )

        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]

        return prompt_cost + completion_cost


class AnthropicProvider(LLMProvider):
    """
    Anthropic LLM provider implementation.

    Supports Claude models.
    """

    def __init__(self, config: LLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. Run: pip install anthropic"
            )

        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=3,
        )

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate text using Anthropic API."""

        # Separate system message
        system_message = None
        user_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                user_messages.append({"role": msg.role, "content": msg.content})

        # Set request parameters
        request_params = {
            "model": self.config.model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        if system_message:
            request_params["system"] = system_message

        try:
            # API call
            response = await self.client.messages.create(**request_params)

            # Convert response
            usage = response.usage

            llm_usage = LLMUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                estimated_cost=self._calculate_cost(
                    usage.input_tokens, usage.output_tokens
                ),
            )

            return LLMResponse(
                content=response.content[0].text,
                usage=llm_usage,
                model=response.model,
                finish_reason=response.stop_reason,
                metadata={"provider": "anthropic"},
            )

        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}") from e

    async def validate_connection(self) -> bool:
        """Validate Anthropic API connection."""
        try:
            test_messages = [LLMMessage(role="user", content="Hello")]
            await self.generate(test_messages, max_tokens=10)
            return True
        except Exception:
            return False

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate Anthropic API cost."""
        # Pricing by model (USD per 1M tokens)
        pricing = {
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
        }

        model_pricing = pricing.get(self.config.model, {"input": 1.0, "output": 5.0})

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Returns predefined responses without actual API calls.
    """

    def __init__(self, config: LLMConfig, mock_responses: list[str] | None = None):
        super().__init__(config)
        self.mock_responses = mock_responses or ["Mock response for testing"]
        self.response_index = 0

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate mock response."""

        # Estimate token count
        prompt_text = " ".join([msg.content for msg in messages])
        prompt_tokens = self.estimate_tokens(prompt_text)

        response_text = self.mock_responses[
            self.response_index % len(self.mock_responses)
        ]
        completion_tokens = self.estimate_tokens(response_text)

        self.response_index += 1

        llm_usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost=0.0,
        )

        return LLMResponse(
            content=response_text,
            usage=llm_usage,
            model="mock-model",
            finish_reason="stop",
            metadata={"provider": "mock", "test_mode": True},
        )

    async def validate_connection(self) -> bool:
        """Always returns True."""
        return True
