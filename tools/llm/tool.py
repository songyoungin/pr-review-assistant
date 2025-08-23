"""
LLM 통합 도구 - NFR-2, NFR-5 준수

이 모듈은 LLM 통합 도구의 핵심 구현을 담당합니다.
다양한 LLM 제공자를 지원하며, 규칙 기반의 코드 분석을 수행합니다.
"""

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, cast

from tools.base import BaseTool, ToolErrorCode, ToolEvidence, ToolResult
from tools.llm.base import LLMConfig, LLMMessage, LLMProvider, LLMResponse
from tools.llm.prompts import PromptManager, get_prompt_manager
from tools.rules import get_rule_registry, validate_evidence_rule_compliance


@dataclass
class LLMAnalysisRequest:
    """
    LLM analysis request structure.

    Contains all information needed for LLM-based analysis.
    """

    analysis_type: str  # 'code_review', 'doc_consistency', 'schema_analysis'
    content: str  # Content to analyze
    context: dict[str, Any]  # Additional context
    files: list[dict[str, Any]]  # File information with paths and content
    metadata: dict[str, Any] | None = None  # Additional metadata


@dataclass
class LLMAnalysisResult:
    """
    LLM analysis result structure with rule compliance (NFR-2, NFR-5).

    Contains structured analysis results with evidence and rule information.
    """

    analysis_type: str
    findings: list[dict[str, Any]]  # Analysis findings with evidence
    summary: str  # Analysis summary
    confidence_score: float  # Confidence score (0.0 to 1.0)
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LLMTool(BaseTool[LLMAnalysisRequest, LLMAnalysisResult]):
    """
    LLM 통합 도구 - NFR-2, NFR-5 준수.

    다양한 LLM 제공자를 통합하고, 규칙 기반의 코드 분석을 수행합니다.
    모든 결과는 근거(파일:라인)와 규칙 ID/버전을 포함합니다.
    """

    def __init__(
        self, provider: LLMProvider, prompt_manager: PromptManager | None = None
    ):
        super().__init__("llm_tool")

        self.provider = provider
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self.rule_registry = get_rule_registry()

    def execute(self, input_data: LLMAnalysisRequest) -> ToolResult[LLMAnalysisResult]:
        """
        Execute LLM analysis with rule compliance.

        Args:
            input_data: Analysis request containing content and context

        Returns:
            Tool result with structured analysis and evidence
        """
        try:
            # Generate prompt using template
            system_prompt, user_prompt = self._generate_prompt(input_data)

            # Create LLM messages
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]

            # Call LLM provider (synchronously)
            llm_response: Any = self.provider.generate(messages)

            # If provider.generate returns a coroutine, run it to completion
            if asyncio.iscoroutine(llm_response):
                llm_response = asyncio.run(llm_response)

            # Ensure the final object is typed as LLMResponse
            llm_response = cast(LLMResponse, llm_response)

            # Parse and validate response
            analysis_result = self._parse_llm_response(
                llm_response.content, input_data.analysis_type, input_data.files
            )

            # Validate evidence rule compliance (NFR-5)
            evidence_list = self._validate_evidence_compliance(
                analysis_result.findings, input_data.files
            )

            # Create tool result
            result = LLMAnalysisResult(
                analysis_type=input_data.analysis_type,
                findings=analysis_result.findings,
                summary=analysis_result.summary,
                confidence_score=analysis_result.confidence_score,
                metadata={
                    "llm_model": llm_response.model,
                    "token_usage": llm_response.usage.to_dict(),
                    "provider": self.provider.provider_name,
                },
            )

            return ToolResult.success(
                output=result,
                evidence=evidence_list,
                metrics=self._create_llm_metrics(llm_response.usage),
            )

        except Exception as e:
            error_message = f"LLM analysis failed: {str(e)}"
            return ToolResult.error(
                error_code=ToolErrorCode.PROCESSING_ERROR,
                error_message=error_message,
                metrics=self._create_metrics(),
            )

    def _generate_prompt(self, request: LLMAnalysisRequest) -> tuple[str, str]:
        """
        Generate system and user prompts based on analysis type.

        Args:
            request: Analysis request

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        template = self.prompt_manager.get_template(request.analysis_type)
        if not template:
            raise ValueError(f"Unknown analysis type: {request.analysis_type}")

        # Prepare template variables
        template_vars = {
            "content": request.content,
            "files": request.files,
            **request.context,
        }

        return template.render(**template_vars)

    def _parse_llm_response(
        self, response_content: str, analysis_type: str, files: list[dict[str, Any]]
    ) -> LLMAnalysisResult:
        """
        Parse LLM response into structured format with evidence.

        Args:
            response_content: Raw LLM response
            analysis_type: Type of analysis performed
            files: File information for evidence validation

        Returns:
            Structured analysis result
        """
        try:
            # Try to extract JSON from response (more flexible pattern)
            json_match = re.search(
                r"```json\s*\n(.*?)\n```", response_content, re.DOTALL | re.IGNORECASE
            )
            if json_match:
                json_str = json_match.group(1).strip()
                parsed_data = json.loads(json_str)
            else:
                # Try to find JSON object in the response
                json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group(0))
                else:
                    # Fallback: try to parse entire response as JSON
                    parsed_data = json.loads(response_content)

            # Extract findings and validate structure
            findings = self._validate_findings_structure(
                parsed_data.get("findings", []), files
            )

            # Calculate confidence score based on evidence quality
            confidence_score = self._calculate_confidence_score(findings)

            return LLMAnalysisResult(
                analysis_type=analysis_type,
                findings=findings,
                summary=parsed_data.get("summary", "Analysis completed"),
                confidence_score=confidence_score,
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e

    def _validate_findings_structure(
        self, findings: list[dict[str, Any]], files: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Validate and enhance findings structure with proper evidence.

        Args:
            findings: Raw findings from LLM
            files: File information for evidence validation

        Returns:
            Validated findings with proper evidence
        """
        validated_findings = []

        for finding in findings:
            # Validate required fields
            if not all(key in finding for key in ["file_path", "rule_id", "severity"]):
                continue  # Skip invalid findings

            # Validate rule ID (NFR-5 compliance)
            rule_id = finding.get("rule_id")
            if rule_id and not validate_evidence_rule_compliance(rule_id):
                # Assign default rule if invalid
                finding["rule_id"] = "QUAL001"  # Default quality rule
                finding["rule_version"] = "1.0.0"

            # Add rule version if missing
            if "rule_version" not in finding:
                rule = self.rule_registry.get_rule(rule_id) if rule_id else None
                finding["rule_version"] = rule.version if rule else "1.0.0"

            # Ensure evidence includes file:line (NFR-2 compliance)
            if "line_number" not in finding:
                finding["line_number"] = 1  # Default line number

            # Validate file path exists in provided files
            file_path = finding.get("file_path")
            if not any(f.get("path") == file_path for f in files):
                continue  # Skip findings for non-existent files

            validated_findings.append(finding)

        return validated_findings

    def _validate_evidence_compliance(
        self, findings: list[dict[str, Any]], files: list[dict[str, Any]]
    ) -> list[ToolEvidence]:
        """
        Create ToolEvidence list with rule compliance (NFR-5).

        Args:
            findings: Validated findings
            files: File information

        Returns:
            List of ToolEvidence objects
        """
        evidence_list = []

        for finding in findings:
            file_path = finding.get("file_path")
            line_number = finding.get("line_number", 1)
            rule_id = finding.get("rule_id")
            rule_version = finding.get("rule_version")

            # Get file content for evidence
            file_content = ""
            for file_info in files:
                if file_info.get("path") == file_path:
                    file_content = file_info.get("content", "")
                    break

            # Create evidence with rule information
            evidence = ToolEvidence(
                file_path=file_path or "",
                content=file_content,
                evidence_type="code_analysis",
                line_number=line_number,
                description=finding.get("message", ""),
                rule_id=rule_id,
                rule_version=rule_version,
            )

            evidence_list.append(evidence)

        return evidence_list

    def _calculate_confidence_score(self, findings: list[dict[str, Any]]) -> float:
        """
        Calculate confidence score based on evidence quality.

        Args:
            findings: Validated findings

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not findings:
            return 0.0

        total_score = 0.0
        for finding in findings:
            score = 0.0

            # File path validation (20%)
            if finding.get("file_path"):
                score += 0.2

            # Line number validation (20%)
            line_num = finding.get("line_number")
            if line_num is not None and line_num > 0:
                score += 0.2

            # Rule ID validation (30%)
            rule_id = finding.get("rule_id")
            if rule_id and validate_evidence_rule_compliance(rule_id):
                score += 0.3

            # Message quality (30%)
            message = finding.get("message", "")
            if len(message.strip()) > 10:  # Meaningful message
                score += 0.3

            total_score += score

        return min(total_score / len(findings), 1.0)

    def _create_llm_metrics(self, usage: Any) -> Any:
        """
        Create tool metrics from LLM usage.

        Args:
            usage: LLM usage information

        Returns:
            Tool metrics object
        """
        from tools.base import ToolMetrics

        return ToolMetrics(
            processing_time_ms=0,  # Will be set by base class
            additional_metrics={
                "llm_tokens_used": usage.total_tokens,
                "llm_prompt_tokens": usage.prompt_tokens,
                "llm_completion_tokens": usage.completion_tokens,
                "llm_cost": usage.estimated_cost,
            },
        )

    def get_supported_analysis_types(self) -> list[str]:
        """Get list of supported analysis types."""
        templates = self.prompt_manager.list_templates()
        return [template.name for template in templates]

    def validate_analysis_request(self, request: LLMAnalysisRequest) -> bool:
        """
        Validate analysis request.

        Args:
            request: Request to validate

        Returns:
            True if valid
        """
        # Check if analysis type is supported
        supported_types = self.get_supported_analysis_types()
        if not supported_types or request.analysis_type not in supported_types:
            return False

        # Check if files are provided
        if not request.files:
            return False

        # Check if content is provided
        if not request.content or not request.content.strip():
            return False

        return True

    def get_analysis_info(self) -> dict[str, Any]:
        """Get tool information and capabilities."""
        return {
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "provider": self.provider.get_provider_info(),
            "supported_analyses": self.get_supported_analysis_types(),
            "rule_compliance": {
                "nfr_2": "Evidence includes file:line references",
                "nfr_5": "Results include rule_id and rule_version",
            },
        }


def create_llm_tool(provider_config: LLMConfig, provider_type: str = "mock") -> LLMTool:
    """
    Factory function to create LLM tool with specified provider.

    Args:
        provider_config: LLM provider configuration
        provider_type: Type of provider ('openai', 'anthropic', 'mock')

    Returns:
        Configured LLM tool instance
    """
    # Resolve provider instance with explicit typing to satisfy mypy unions
    provider: LLMProvider
    try:
        if provider_type == "openai":
            from .providers import OpenAIProvider

            provider = OpenAIProvider(provider_config)
        elif provider_type == "anthropic":
            from .providers import AnthropicProvider

            provider = AnthropicProvider(provider_config)
        elif provider_type == "mock":
            from .providers import MockProvider

            provider = MockProvider(provider_config)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    except ImportError:
        # Fallback to mock provider if dependencies are not available
        from .providers import MockProvider

        provider = MockProvider(provider_config)

    return LLMTool(provider=provider)


# Export key functions and classes
__all__ = ["LLMTool", "LLMAnalysisRequest", "LLMAnalysisResult", "create_llm_tool"]
