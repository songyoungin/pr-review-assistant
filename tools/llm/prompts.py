"""
Prompt management system.

Manages template-based prompts for LLM requests and handles generation.
Supports rule information inclusion for NFR-5 compliance.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from tools.rules import get_rule_registry


@dataclass
class PromptTemplate:
    """
    Prompt template structure.

    Defines reusable prompt templates.
    """

    name: str  # Template name
    description: str  # Template description
    system_prompt: str  # System prompt
    user_prompt_template: str  # User prompt template
    version: str = "1.0.0"  # Template version
    variables: list[str] | None = None  # Template variables list
    tags: list[str] | None = None  # Tags list
    created_at: str | None = None  # Creation date
    updated_at: str | None = None  # Last modified date

    def __post_init__(self) -> None:
        if self.variables is None:
            self.variables = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now(UTC).isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now(UTC).isoformat()

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """
        Render template into system prompt and user prompt.

        Args:
            **kwargs: Template variable values

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        try:
            user_prompt = self.user_prompt_template.format(**kwargs)
            return self.system_prompt, user_prompt
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}") from e

    def validate_variables(self, variables: dict[str, Any]) -> bool:
        """
        Validate if provided variables meet template requirements.

        Args:
            variables: Variables to validate

        Returns:
            True if valid
        """
        return all(var in variables for var in (self.variables or []))

    def to_dict(self) -> dict[str, Any]:
        """Convert template to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """Create template from dictionary."""
        return cls(**data)


class PromptManager:
    """
    Prompt template manager.

    Manages prompt templates for various analysis purposes.
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._rule_registry = get_rule_registry()
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default prompt templates."""

        # Code review template
        code_review_template = PromptTemplate(
            name="code_review",
            description="Prompt for code quality, security, and performance analysis",
            system_prompt="""당신은 PR 리뷰를 전문으로 하는 코드 분석가입니다.

## 역할 및 책임
- 코드 품질, 보안, 성능을 종합적으로 분석
- NFR-2: 모든 지적에는 최소 1개 이상의 근거(파일:라인) 포함
- NFR-5: 각 결과는 규칙 ID와 버전을 포함하여 추적 가능하도록 함

## 핵심 원칙
1. **증거 기반 판단**: 변경사항(diff, 파일:라인)을 기반으로만 판단
2. **결정적 출력**: temperature=0으로 일관된 결과 제공
3. **규칙 준수**: 모든 결과에 rule_id와 rule_version 명시
4. **근거 필수**: 모든 지적에 구체적인 파일 경로와 라인 번호 포함

## 출력 형식
JSON 형식으로 다음 구조를 따르세요:
```json
{
  "findings": [
    {
      "file_path": "string",
      "line_number": "number",
      "rule_id": "string",
      "rule_version": "string",
      "severity": "high|medium|low",
      "category": "security|performance|quality",
      "message": "string",
      "evidence": "string",
      "suggestion": "string"
    }
  ]
}
```

## 규칙 ID 예시
- SEC001: SQL Injection 취약점
- PERF001: N+1 쿼리 패턴
- QUAL001: 타입 힌트 누락""",
            user_prompt_template="""다음 코드 변경사항을 분석하여 코드 품질, 보안, 성능 이슈를 찾아주세요:

## 변경 파일들
{changed_files}

## 주요 변경사항
{diff_content}

## 분석 요청
1. **보안 취약점**: SQL Injection, XSS, 인증 우회 등
2. **성능 문제**: N+1 쿼리, 비효율적 알고리즘, 메모리 누수
3. **코드 품질**: 타입 힌트 누락, 복잡도, 유지보수성

각 발견사항은 다음을 포함해야 합니다:
- 정확한 파일 경로와 라인 번호
- 적용된 규칙 ID와 버전
- 심각도 (high/medium/low)
- 구체적인 문제 설명과 개선 제안""",
            variables=["changed_files", "diff_content"],
            tags=["code_review", "security", "performance", "quality"],
        )

        # 문서 일관성 템플릿
        doc_consistency_template = PromptTemplate(
            name="doc_consistency",
            description="API 변경과 문서 일치성 검사",
            system_prompt="""당신은 문서 품질 관리 전문가입니다.

## 역할 및 책임
- API 변경사항과 문서의 일치성 자동 검증
- 누락된 문서나 오래된 문서 자동 탐지
- NFR-2: 모든 지적에 근거(파일:라인) 포함
- NFR-5: 규칙 ID와 버전으로 결과 추적

## 출력 형식
```json
{
  "mismatches": [
    {
      "file_path": "string",
      "line_number": "number",
      "rule_id": "string",
      "rule_version": "string",
      "type": "missing|outdated|inconsistent",
      "message": "string",
      "suggestion": "string"
    }
  ]
}
```""",
            user_prompt_template="""다음 API 변경사항을 문서와 비교하여 일관성을 검증해주세요:

## API 변경사항
{api_changes}

## 문서 파일들
{doc_files}

## 검증 요청
1. 새로운 API 엔드포인트의 문서화 여부
2. 기존 API 변경사항의 문서 업데이트 필요성
3. 문서와 코드 간 불일치 사항""",
            variables=["api_changes", "doc_files"],
            tags=["documentation", "api", "consistency"],
        )

        # 스키마 분석 템플릿
        schema_analysis_template = PromptTemplate(
            name="schema_analysis",
            description="데이터베이스 스키마 변경 영향 분석",
            system_prompt="""당신은 데이터베이스 스키마 전문가입니다.

## 역할 및 책임
- 스키마 변경의 브레이킹 체인지 분석
- 마이그레이션 전략 및 운영 가이드 생성
- NFR-2, NFR-5 준수

## 출력 형식
```json
{
  "breaking_changes": [
    {
      "file_path": "string",
      "line_number": "number",
      "rule_id": "string",
      "rule_version": "string",
      "type": "table_removal|column_removal|type_change",
      "message": "string",
      "migration_strategy": "string"
    }
  ]
}
```""",
            user_prompt_template="""다음 스키마 변경사항을 분석하여 브레이킹 체인지를 찾아주세요:

## 스키마 변경사항
{schema_changes}

## 분석 요청
1. 롤백 불가능한 변경사항 식별
2. 애플리케이션 호환성 영향 평가
3. 마이그레이션 전략 제안""",
            variables=["schema_changes"],
            tags=["schema", "database", "migration"],
        )

        # 템플릿 등록
        self._templates = {
            "code_review": code_review_template,
            "doc_consistency": doc_consistency_template,
            "schema_analysis": schema_analysis_template,
        }

    def register_template(self, template: PromptTemplate) -> None:
        """새로운 프롬프트 템플릿을 등록합니다."""
        if template.name in self._templates:
            raise ValueError(f"Template {template.name} already exists")
        self._templates[template.name] = template

    def get_template(self, name: str) -> PromptTemplate | None:
        """이름으로 프롬프트 템플릿을 조회합니다."""
        return self._templates.get(name)

    def list_templates(self, tag: str | None = None) -> list[PromptTemplate]:
        """등록된 템플릿들을 반환합니다."""
        if tag:
            return [t for t in self._templates.values() if tag in (t.tags or [])]
        return list(self._templates.values())

    def render_template(self, name: str, **kwargs: Any) -> tuple[str, str]:
        """
        템플릿을 렌더링합니다.

        Args:
            name: 템플릿 이름
            **kwargs: 템플릿 변수들

        Returns:
            (system_prompt, user_prompt) 튜플
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template {name} not found")

        if not template.validate_variables(kwargs):
            missing = set(template.variables or []) - set(kwargs.keys())
            raise ValueError(f"Missing template variables: {missing}")

        return template.render(**kwargs)

    def get_template_info(self, name: str) -> dict[str, Any] | None:
        """템플릿 정보를 반환합니다."""
        template = self.get_template(name)
        if template:
            return template.to_dict()
        return None

    def export_templates(self) -> str:
        """모든 템플릿을 JSON으로 내보냅니다."""
        templates_dict = {
            name: template.to_dict() for name, template in self._templates.items()
        }
        return json.dumps(templates_dict, indent=2, ensure_ascii=False)

    def import_templates(self, json_str: str) -> None:
        """JSON으로부터 템플릿들을 불러옵니다."""
        templates_dict = json.loads(json_str)
        for name, template_data in templates_dict.items():
            template = PromptTemplate.from_dict(template_data)
            self._templates[name] = template

    def get_available_analyses(self) -> list[dict[str, Any]]:
        """
        사용 가능한 분석 유형들을 반환합니다.

        Returns:
            분석 유형 정보 목록
        """
        return [
            {
                "name": template.name,
                "description": template.description,
                "tags": template.tags,
                "variables": template.variables,
            }
            for template in self._templates.values()
        ]


# 전역 프롬프트 매니저 인스턴스
prompt_manager = PromptManager()


def get_prompt_manager() -> PromptManager:
    """전역 프롬프트 매니저를 반환합니다."""
    return prompt_manager
