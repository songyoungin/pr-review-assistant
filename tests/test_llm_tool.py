"""
LLM integration tool test - NFR-2, NFR-5 compliance verification

This module tests the functionality of LLM integration tools.
It specifically verifies NFR-2 (evidence inclusion) and NFR-5 (rule ID/version) compliance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from tools.base import ToolResult, ToolStatus
from tools.llm.base import LLMConfig, LLMResponse, LLMUsage
from tools.llm.providers import MockProvider
from tools.llm.tool import (
    LLMAnalysisRequest,
    LLMAnalysisResult,
    LLMTool,
    create_llm_tool,
)


class TestLLMTool:
    """Test class for LLM integration tools."""

    @pytest.fixture
    def mock_provider(self) -> Any:
        """Mock LLM provider fixture."""
        config = LLMConfig(api_key="test-key", model="test-model")
        return MockProvider(config)

    @pytest.fixture
    def llm_tool(self: TestLLMTool, mock_provider: Any) -> LLMTool:
        """LLM tool fixture."""
        return LLMTool(provider=mock_provider)

    @pytest.fixture
    def sample_request(self) -> Any:
        """Sample analysis request fixture."""
        return LLMAnalysisRequest(
            analysis_type="code_review",
            content="def test_function() -> None: pass",
            context={
                "language": "python",
                "changed_files": ["test.py"],
                "diff_content": "def test_function() -> None: pass",
            },
            files=[
                {
                    "path": "test.py",
                    "content": "def test_function() -> None: pass",
                    "language": "python",
                }
            ],
        )

    @pytest.fixture
    def mock_llm_response(self) -> Any:
        """Mock LLM response fixture."""
        return LLMResponse(
            content='```json\n{"findings": [{"file_path": "test.py", "line_number": 1, "rule_id": "QUAL001", "severity": "low", "message": "Test finding"}], "summary": "Test analysis completed"}\n```',
            usage=LLMUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                estimated_cost=0.002,
            ),
            model="test-model",
            finish_reason="stop",
        )

    def test_tool_initialization(self, llm_tool: LLMTool) -> None:
        """Test LLM tool initialization."""
        assert llm_tool.tool_name == "llm_tool"
        assert llm_tool.provider is not None
        assert llm_tool.prompt_manager is not None
        assert llm_tool.rule_registry is not None

    def test_supported_analysis_types(self, llm_tool: LLMTool) -> None:
        """Test getting supported analysis types."""
        types = llm_tool.get_supported_analysis_types()
        assert isinstance(types, list)
        assert len(types) > 0

    def test_validate_analysis_request_valid(
        self, llm_tool: LLMTool, sample_request: LLMAnalysisRequest
    ) -> None:
        """Test validation of valid analysis request."""
        assert llm_tool.validate_analysis_request(sample_request) is True

    def test_validate_analysis_request_invalid(self, llm_tool: LLMTool) -> None:
        """Test validation of invalid analysis request."""
        # Invalid analysis type
        invalid_request = LLMAnalysisRequest(
            analysis_type="invalid_type",
            content="test content",
            context={},
            files=[{"path": "test.py", "content": "test"}],
        )
        assert llm_tool.validate_analysis_request(invalid_request) is False

        # No files
        no_files_request = LLMAnalysisRequest(
            analysis_type="code_review", content="test content", context={}, files=[]
        )
        assert llm_tool.validate_analysis_request(no_files_request) is False

        # Empty content
        empty_content_request = LLMAnalysisRequest(
            analysis_type="code_review",
            content="",
            context={},
            files=[{"path": "test.py", "content": "test"}],
        )
        assert llm_tool.validate_analysis_request(empty_content_request) is False

    def test_execute_success(
        self,
        llm_tool: LLMTool,
        sample_request: LLMAnalysisRequest,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test successful execution."""
        # Mock the provider's generate method
        provider_obj: Any = llm_tool.provider
        if callable(provider_obj):
            provider_obj = provider_obj()
        provider_obj.generate = Mock(return_value=mock_llm_response)

        result = llm_tool.execute(sample_request)

        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.output is not None
        assert isinstance(result.output, LLMAnalysisResult)
        assert result.output.analysis_type == "code_review"
        assert len(result.evidence) > 0

    def test_execute_with_invalid_response(
        self, llm_tool: LLMTool, sample_request: LLMAnalysisRequest
    ) -> None:
        """Test execution with invalid LLM response."""
        # Mock invalid JSON response
        invalid_response = LLMResponse(
            content="Invalid JSON response",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="test-model",
            finish_reason="stop",
        )

        provider_obj: Any = llm_tool.provider
        if callable(provider_obj):
            provider_obj = provider_obj()
        provider_obj.generate = Mock(return_value=invalid_response)

        result = llm_tool.execute(sample_request)

        assert result.status == ToolStatus.ERROR
        assert result.error_code is not None

    def test_nfr_2_compliance_evidence_includes_file_line(
        self, llm_tool: LLMTool
    ) -> None:
        """Test NFR-2 compliance: Evidence includes file:line references."""
        findings = [
            {
                "file_path": "test.py",
                "line_number": 1,
                "rule_id": "QUAL001",
                "severity": "low",
                "message": "Test finding",
            }
        ]

        files = [{"path": "test.py", "content": "test content"}]

        evidence_list = llm_tool._validate_evidence_compliance(findings, files)

        assert len(evidence_list) == 1
        evidence = evidence_list[0]
        assert evidence.file_path == "test.py"
        assert evidence.line_number == 1
        assert evidence.content == "test content"

    def test_nfr_5_compliance_rule_id_and_version(self, llm_tool: LLMTool) -> None:
        """Test NFR-5 compliance: Results include rule_id and rule_version."""
        findings = [
            {
                "file_path": "test.py",
                "line_number": 1,
                "rule_id": "QUAL001",
                "severity": "low",
                "message": "Test finding",
            }
        ]

        files = [{"path": "test.py", "content": "test content"}]

        # Validate findings structure
        validated_findings = llm_tool._validate_findings_structure(findings, files)

        assert len(validated_findings) == 1
        finding = validated_findings[0]
        assert "rule_id" in finding
        assert "rule_version" in finding
        assert finding["rule_id"] == "QUAL001"
        assert finding["rule_version"] == "1.0.0"  # Default version

    def test_invalid_rule_id_handling(self, llm_tool: LLMTool) -> None:
        """Test handling of invalid rule IDs."""
        findings = [
            {
                "file_path": "test.py",
                "line_number": 1,
                "rule_id": "INVALID_RULE",
                "severity": "low",
                "message": "Test finding",
            }
        ]

        files = [{"path": "test.py", "content": "test content"}]

        validated_findings = llm_tool._validate_findings_structure(findings, files)

        assert len(validated_findings) == 1
        finding = validated_findings[0]
        # Should be replaced with default rule
        assert finding["rule_id"] == "QUAL001"
        assert finding["rule_version"] == "1.0.0"

    def test_confidence_score_calculation(self, llm_tool: LLMTool) -> None:
        """Test confidence score calculation."""
        # High quality finding
        good_findings = [
            {
                "file_path": "test.py",
                "line_number": 1,
                "rule_id": "QUAL001",
                "severity": "low",
                "message": "This is a meaningful message with sufficient detail",
            }
        ]

        # Low quality finding
        bad_findings = [
            {
                "file_path": "test.py",
                "line_number": None,
                "rule_id": "INVALID",
                "severity": "low",
                "message": "Bad",
            }
        ]

        good_score = llm_tool._calculate_confidence_score(good_findings)
        bad_score = llm_tool._calculate_confidence_score(bad_findings)

        assert good_score > bad_score
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= bad_score <= 1.0

    def test_parse_llm_response_json_extraction(self, llm_tool: LLMTool) -> None:
        """Test JSON extraction from LLM response."""
        json_response = """
        Here is my analysis:

        ```json
        {
            "findings": [
                {
                    "file_path": "test.py",
                    "line_number": 1,
                    "rule_id": "QUAL001",
                    "severity": "low",
                    "message": "Test finding"
                }
            ],
            "summary": "Test analysis completed"
        }
        ```
        """

        files = [{"path": "test.py", "content": "test content"}]

        result = llm_tool._parse_llm_response(json_response, "code_review", files)

        assert isinstance(result, LLMAnalysisResult)
        assert result.analysis_type == "code_review"
        assert len(result.findings) == 1
        assert result.summary == "Test analysis completed"

    def test_get_analysis_info(self, llm_tool: LLMTool) -> None:
        """Test getting analysis information."""
        info = llm_tool.get_analysis_info()

        assert "tool_name" in info
        assert "tool_id" in info
        assert "provider" in info
        assert "supported_analyses" in info
        assert "rule_compliance" in info

        # Check NFR compliance info
        compliance = info["rule_compliance"]
        assert "nfr_2" in compliance
        assert "nfr_5" in compliance


class TestLLMToolFactory:
    """Test LLM tool factory function."""

    def test_create_openai_tool(self) -> None:
        """Test creating OpenAI tool."""
        config = LLMConfig(api_key="test-key", model="gpt-3.5-turbo")

        tool = create_llm_tool(config, "openai")
        assert isinstance(tool, LLMTool)
        # Falls back to mock provider when dependencies are not available
        assert tool.provider.provider_name == "MockProvider"

    def test_create_anthropic_tool(self) -> None:
        """Test creating Anthropic tool."""
        config = LLMConfig(api_key="test-key", model="claude-3-sonnet")

        tool = create_llm_tool(config, "anthropic")
        assert isinstance(tool, LLMTool)
        # Falls back to mock provider when dependencies are not available
        assert tool.provider.provider_name == "MockProvider"

    def test_create_mock_tool(self) -> None:
        """Test creating mock tool."""
        config = LLMConfig(api_key="test-key", model="mock-model")

        tool = create_llm_tool(config, "mock")
        assert isinstance(tool, LLMTool)
        assert tool.provider.provider_name == "MockProvider"

    def test_create_invalid_provider(self) -> None:
        """Test creating tool with invalid provider type."""
        config = LLMConfig(api_key="test-key", model="test-model")

        with pytest.raises(ValueError, match="Unsupported provider type"):
            create_llm_tool(config, "invalid_provider")


class TestLLMAnalysisResult:
    """Test LLMAnalysisResult class."""

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        result = LLMAnalysisResult(
            analysis_type="code_review",
            findings=[
                {
                    "file_path": "test.py",
                    "line_number": 1,
                    "rule_id": "QUAL001",
                    "severity": "low",
                    "message": "Test finding",
                }
            ],
            summary="Test analysis",
            confidence_score=0.8,
        )

        result_dict = result.to_dict()

        assert result_dict["analysis_type"] == "code_review"
        assert len(result_dict["findings"]) == 1
        assert result_dict["summary"] == "Test analysis"
        assert result_dict["confidence_score"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
