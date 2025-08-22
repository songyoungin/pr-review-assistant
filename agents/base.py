"""
Base Agent Contracts and Data Structures

This module defines the core contracts and data structures for the multi-agent
PR review assistant system. All agents must adhere to these contracts for
consistent communication and state management.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AgentStatus(Enum):
    """Status values for agent responses."""

    OK = "ok"
    ERROR = "error"
    PARTIAL = "partial"


class ErrorCode(Enum):
    """Standard error codes for agent operations."""

    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    TIMEOUT = "TIMEOUT"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    PERMISSION_ERROR = "PERMISSION_ERROR"


@dataclass
class RequestEnvelope:
    """
    Standard request envelope for agent communication.

    All requests between agents must use this envelope format.
    """

    request_id: str
    created_at: str  # ISO 8601 format
    agent: str  # Agent name/type
    payload: dict[str, Any]
    attachments: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Validate and set defaults after initialization."""
        if not self.request_id:
            self.request_id = str(uuid.uuid4())

        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequestEnvelope":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "RequestEnvelope":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Evidence:
    """
    Evidence structure for supporting agent claims and findings.

    All agent outputs must include evidence with specific file:line references.
    """

    file_path: str
    content: str  # The actual evidence content
    evidence_type: str  # 'code', 'doc', 'config', 'log', etc.
    line_number: int | None = None
    description: str | None = None  # Human-readable explanation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Metrics:
    """
    Performance and quality metrics for agent operations.
    """

    processing_time_ms: int
    tokens_used: int | None = None  # For LLM operations
    files_processed: int | None = None
    lines_processed: int | None = None
    memory_usage_mb: float | None = None
    additional_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ResponseEnvelope:
    """
    Standard response envelope for agent communication.

    All responses from agents must use this envelope format.
    """

    request_id: str
    agent: str
    status: AgentStatus
    payload: dict[str, Any]
    evidence: list[Evidence]
    metrics: Metrics
    error_code: ErrorCode | None = None
    error_message: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        """Validate response data."""
        if self.status == AgentStatus.ERROR and not self.error_code:
            raise ValueError("error_code is required when status is ERROR")

        if not self.evidence:
            raise ValueError("At least one evidence item is required")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        if self.error_code:
            result["error_code"] = self.error_code.value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponseEnvelope":
        """Create from dictionary."""
        # Convert string values back to enums
        if "status" in data:
            data["status"] = AgentStatus(data["status"])
        if "error_code" in data and data["error_code"]:
            data["error_code"] = ErrorCode(data["error_code"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ResponseEnvelope":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class RepositoryInfo:
    """Repository information structure."""

    provider: str  # 'github', 'gitlab', 'bitbucket', etc.
    url: str
    default_branch: str
    owner: str | None = None
    repo_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PullRequestInfo:
    """Pull request information structure."""

    number: int
    title: str
    base: str  # base branch
    head: str  # head branch
    author: str
    url: str
    created_at: str | None = None
    updated_at: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DiffInfo:
    """Diff information structure."""

    unified_patch_path: str  # Path to the unified diff file
    changed_files_path: str  # Path to the changed files list
    total_files: int | None = None
    total_additions: int | None = None
    total_deletions: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AgentOutputs:
    """Agent output tracking structure."""

    summary: str | None = None  # Path to summary output
    code_review: str | None = None  # Path to code review output
    docs: str | None = None  # Path to documentation analysis output
    schema: str | None = None  # Path to schema analysis output
    final_report: str | None = None  # Path to final report

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OrchestratorState:
    """
    Central state management for the orchestrator.

    This maintains the current state of the entire PR review process
    and tracks outputs from all agents.
    """

    repo: RepositoryInfo
    diff: DiffInfo
    pr: PullRequestInfo | None = None
    outputs: AgentOutputs = field(default_factory=AgentOutputs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert nested dataclasses to dicts
        result["repo"] = self.repo.to_dict()
        if self.pr:
            result["pr"] = self.pr.to_dict()
        result["diff"] = self.diff.to_dict()
        result["outputs"] = self.outputs.to_dict()
        return result

    def to_json(self, file_path: str | Path | None = None) -> str:
        """Convert to JSON string and optionally save to file."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)

        if file_path:
            Path(file_path).write_text(json_str, encoding="utf-8")

        return json_str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestratorState":
        """Create from dictionary."""
        # Convert nested dicts back to dataclasses
        data["repo"] = RepositoryInfo(**data["repo"])
        if data.get("pr"):
            data["pr"] = PullRequestInfo(**data["pr"])
        data["diff"] = DiffInfo(**data["diff"])
        data["outputs"] = AgentOutputs(**data["outputs"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "OrchestratorState":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, file_path: str | Path) -> "OrchestratorState":
        """Load from JSON file."""
        return cls.from_json(Path(file_path).read_text(encoding="utf-8"))


# Base Agent Interface
class BaseAgent:
    """
    Base interface for all agents in the system.

    All agents must implement this interface to ensure consistent
    communication and behavior.
    """

    def __init__(self, agent_name: str) -> None:
        """Initialize the agent with a unique name."""
        self.agent_name = agent_name
        self.agent_id = f"{agent_name}_{uuid.uuid4().hex[:8]}"

    def process_request(self, request: RequestEnvelope) -> ResponseEnvelope:
        """
        Process a request and return a response.

        Args:
            request: The request envelope to process

        Returns:
            Response envelope with results or error information

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process_request")

    def create_success_response(
        self,
        request_id: str,
        payload: dict[str, Any],
        evidence: list[Evidence],
        metrics: Metrics,
    ) -> ResponseEnvelope:
        """Create a successful response envelope."""
        return ResponseEnvelope(
            request_id=request_id,
            agent=self.agent_name,
            status=AgentStatus.OK,
            payload=payload,
            evidence=evidence,
            metrics=metrics,
        )

    def create_error_response(
        self,
        request_id: str,
        error_code: ErrorCode,
        error_message: str,
        evidence: list[Evidence],
        metrics: Metrics,
        retryable: bool = False,
    ) -> ResponseEnvelope:
        """Create an error response envelope."""
        return ResponseEnvelope(
            request_id=request_id,
            agent=self.agent_name,
            status=AgentStatus.ERROR,
            payload={},
            evidence=evidence,
            metrics=metrics,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
        )

    def create_partial_response(
        self,
        request_id: str,
        payload: dict[str, Any],
        evidence: list[Evidence],
        metrics: Metrics,
    ) -> ResponseEnvelope:
        """Create a partial response envelope."""
        return ResponseEnvelope(
            request_id=request_id,
            agent=self.agent_name,
            status=AgentStatus.PARTIAL,
            payload=payload,
            evidence=evidence,
            metrics=metrics,
        )
