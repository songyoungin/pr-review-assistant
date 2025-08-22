"""
Tools Base Interface

This module defines the common interface that all tools in the PR Review Assistant must follow.
All tools inherit from this base class to ensure consistent behavior and error handling.
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from loguru import logger

# Generic type for tool input/output
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ToolStatus(Enum):
    """Tool execution status values."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class ToolErrorCode(Enum):
    """Standard tool error codes."""

    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    TIMEOUT = "TIMEOUT"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"


@dataclass
class ToolMetrics:
    """Tool execution performance and quality metrics."""

    processing_time_ms: int
    files_processed: int | None = None
    lines_processed: int | None = None
    memory_usage_mb: float | None = None
    additional_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return asdict(self)


@dataclass
class ToolEvidence:
    """Evidence structure supporting tool results."""

    file_path: str
    content: str  # The actual evidence content
    evidence_type: str  # 'code', 'doc', 'config', 'log', etc.
    line_number: int | None = None
    description: str | None = None  # Human-readable explanation

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return asdict(self)


@dataclass
class ToolResult[OutputT]:
    """Standard structure for tool execution results."""

    status: ToolStatus
    output: OutputT | None = None
    error_code: ToolErrorCode | None = None
    error_message: str | None = None
    evidence: list[ToolEvidence] = field(default_factory=list)
    metrics: ToolMetrics | None = None
    warnings: list[str] = field(default_factory=list)
    retryable: bool = False

    def __post_init__(self) -> None:
        """Validate result data."""
        if self.status == ToolStatus.ERROR and not self.error_code:
            raise ValueError("error_code is required when status is ERROR")

        if self.status == ToolStatus.SUCCESS and not self.output:
            logger.warning("Status is SUCCESS but no output provided")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        if self.error_code:
            result["error_code"] = self.error_code.value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def success(
        cls,
        output: OutputT,
        evidence: list[ToolEvidence] | None = None,
        metrics: ToolMetrics | None = None,
        warnings: list[str] | None = None,
    ) -> "ToolResult[OutputT]":
        """Create success result."""
        return cls(
            status=ToolStatus.SUCCESS,
            output=output,
            evidence=evidence or [],
            metrics=metrics,
            warnings=warnings or [],
        )

    @classmethod
    def error(
        cls,
        error_code: ToolErrorCode,
        error_message: str,
        evidence: list[ToolEvidence] | None = None,
        metrics: ToolMetrics | None = None,
        retryable: bool = False,
    ) -> "ToolResult[OutputT]":
        """Create error result."""
        return cls(
            status=ToolStatus.ERROR,
            error_code=error_code,
            error_message=error_message,
            evidence=evidence or [],
            metrics=metrics,
            retryable=retryable,
        )

    @classmethod
    def partial(
        cls,
        output: OutputT,
        evidence: list[ToolEvidence] | None = None,
        metrics: ToolMetrics | None = None,
        warnings: list[str] | None = None,
    ) -> "ToolResult[OutputT]":
        """Create partial success result."""
        return cls(
            status=ToolStatus.PARTIAL,
            output=output,
            evidence=evidence or [],
            metrics=metrics,
            warnings=warnings or [],
        )


class BaseTool[InputT, OutputT](ABC):
    """
    Base class for all tools.

    All tools that inherit from this class ensure consistent interface and error handling.
    """

    def __init__(self, tool_name: str) -> None:
        """Initialize the tool."""
        self.tool_name = tool_name
        self.tool_id = f"{tool_name}_{uuid.uuid4().hex[:8]}"
        self.start_time: datetime | None = None

    @abstractmethod
    def execute(self, input_data: InputT) -> ToolResult[OutputT]:
        """
        Main method for tool execution.

        Args:
            input_data: Input data required for tool execution

        Returns:
            Tool execution result
        """
        pass

    def _start_execution(self) -> None:
        """Record execution start time."""
        self.start_time = datetime.now(UTC)

    def _end_execution(self) -> int:
        """Calculate execution end and processing time (milliseconds)."""
        if not self.start_time:
            return 0

        end_time = datetime.now(UTC)
        duration = (end_time - self.start_time).total_seconds() * 1000
        return int(duration)

    def _create_metrics(self, **kwargs: Any) -> ToolMetrics:
        """Create metrics object."""
        processing_time = self._end_execution()
        return ToolMetrics(processing_time_ms=processing_time, **kwargs)

    def _log_execution_start(self, input_data: InputT) -> None:
        """Log execution start."""
        logger.info(f"Tool {self.tool_name} execution started: {self.tool_id}")
        if hasattr(input_data, "__dict__"):
            logger.debug(f"Input data: {input_data}")

    def _log_execution_success(self, result: ToolResult[OutputT]) -> None:
        """Log execution success."""
        logger.info(f"Tool {self.tool_name} execution successful: {self.tool_id}")
        if result.metrics:
            logger.debug(f"Processing time: {result.metrics.processing_time_ms}ms")

    def _log_execution_error(self, error: Exception, error_code: ToolErrorCode) -> None:
        """Log execution error."""
        logger.error(f"Tool {self.tool_name} execution failed: {self.tool_id}")
        logger.error(f"Error code: {error_code.value}")
        logger.error(f"Error message: {str(error)}")

    def run(self, input_data: InputT) -> ToolResult[OutputT]:
        """
        Wrapper method for tool execution.

        This method handles common logging, error handling, and metrics collection.

        Args:
            input_data: Input data required for tool execution

        Returns:
            Tool execution result
        """
        try:
            self._start_execution()
            self._log_execution_start(input_data)

            # Execute the actual tool
            result = self.execute(input_data)

            # Add metrics if not present
            if not result.metrics:
                result.metrics = self._create_metrics()

            self._log_execution_success(result)
            return result

        except Exception as e:
            # Handle errors
            error_code = self._classify_error(e)
            error_result: ToolResult[OutputT] = ToolResult.error(
                error_code=error_code,
                error_message=str(e),
                metrics=self._create_metrics(),
                retryable=self._is_retryable_error(error_code),
            )

            self._log_execution_error(e, error_code)
            return error_result

    def _classify_error(self, error: Exception) -> ToolErrorCode:
        """Classify error into standard error codes."""

        if isinstance(error, ValueError | TypeError):
            return ToolErrorCode.INVALID_INPUT
        elif isinstance(error, FileNotFoundError):
            return ToolErrorCode.FILE_NOT_FOUND
        elif isinstance(error, PermissionError):
            return ToolErrorCode.PERMISSION_ERROR
        elif isinstance(error, TimeoutError):
            return ToolErrorCode.TIMEOUT
        else:
            return ToolErrorCode.PROCESSING_ERROR

    def _is_retryable_error(self, error_code: ToolErrorCode) -> bool:
        """Determine if error is retryable."""
        return error_code in {
            ToolErrorCode.TIMEOUT,
            ToolErrorCode.DEPENDENCY_ERROR,
            ToolErrorCode.NETWORK_ERROR,
        }

    def validate_input(self, input_data: InputT) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input data to validate

        Returns:
            Whether validation passed
        """
        # Default implementation: always return True
        # Override in subclasses if needed
        return True

    def get_tool_info(self) -> dict[str, Any]:
        """Return tool information."""
        return {
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "tool_type": self.__class__.__name__,
            "description": getattr(self, "__doc__", "No description"),
        }
