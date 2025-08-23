"""
Rule management system for PR Review Assistant.

This module manages all rules used in the PR review system with their IDs and versions
for traceability (NFR-5 compliance). Each rule has a unique ID and version information
enabling change tracking.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RuleCategory(Enum):
    """
    Enumeration of rule categories.

    Attributes:
        SECURITY: Security-related rules
        PERFORMANCE: Performance optimization rules
        QUALITY: Code quality rules
        STYLE: Code style rules
        DOCUMENTATION: Documentation rules
        SCHEMA: Database schema rules
    """

    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    SCHEMA = "schema"


class Severity(Enum):
    """
    Enumeration of rule severity levels.

    Attributes:
        CRITICAL: Critical severity issues
        HIGH: High severity issues
        MEDIUM: Medium severity issues
        LOW: Low severity issues
        INFO: Informational issues
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class RuleDefinition:
    """
    규칙 정의 구조.

    NFR-5 준수를 위한 규칙 메타데이터를 포함합니다.
    """

    rule_id: str  # 고유 규칙 ID (예: "SEC001", "PERF001")
    name: str  # 규칙 이름
    category: RuleCategory  # 규칙 카테고리
    severity: Severity  # 기본 심각도
    version: str  # 규칙 버전 (예: "1.0.0")
    description: str  # 규칙 설명
    examples: list[str]  # 적용 예시
    created_at: str  # 생성일
    updated_at: str  # 최종 수정일
    deprecated: bool = False  # 폐기 여부
    replacement_rule: str | None = None  # 대체 규칙 ID

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "category": self.category.value,
            "severity": self.severity.value,
            "version": self.version,
            "description": self.description,
            "examples": self.examples,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deprecated": self.deprecated,
            "replacement_rule": self.replacement_rule,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleDefinition":
        """딕셔너리로부터 생성."""
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            category=RuleCategory(data["category"]),
            severity=Severity(data["severity"]),
            version=data["version"],
            description=data["description"],
            examples=data["examples"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            deprecated=data.get("deprecated", False),
            replacement_rule=data.get("replacement_rule"),
        )


class RuleRegistry:
    """
    규칙 레지스트리 - 중앙화된 규칙 관리.

    모든 규칙의 등록, 조회, 버전 관리를 담당합니다.
    """

    def __init__(self) -> None:
        self._rules: dict[str, RuleDefinition] = {}
        self._load_default_rules()

    def _load_default_rules(self) -> None:
        """Load default rules."""
        default_rules = [
            # Security rules (NFR-5 example)
            RuleDefinition(
                rule_id="SEC001",
                name="SQL Injection 취약점",
                category=RuleCategory.SECURITY,
                severity=Severity.HIGH,
                version="1.0.0",
                description="SQL 쿼리에서 사용자 입력을 직접 삽입하는 보안 취약점",
                examples=[
                    'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")',
                    "query = f\"DELETE FROM table WHERE col = '{value}'\"",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="SEC002",
                name="XSS 취약점",
                category=RuleCategory.SECURITY,
                severity=Severity.HIGH,
                version="1.0.0",
                description="크로스 사이트 스크립팅 취약점 - 사용자 입력을 HTML에 직접 출력",
                examples=[
                    'return f"<div>{user_input}</div>"',
                    "response.write(request.GET['param'])",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="PERF001",
                name="N+1 쿼리 패턴",
                category=RuleCategory.PERFORMANCE,
                severity=Severity.MEDIUM,
                version="1.0.0",
                description="루프 내에서 개별 쿼리를 실행하는 비효율적인 패턴",
                examples=[
                    "for user in users: user_data = db.query(UserData).filter(id=user.id).first()",
                    "for item in items: category = Category.query.get(item.category_id)",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="QUAL001",
                name="타입 힌트 누락",
                category=RuleCategory.QUALITY,
                severity=Severity.LOW,
                version="1.0.0",
                description="함수 매개변수나 반환값에 타입 힌트를 제공하지 않음",
                examples=[
                    "def process_data(data): return data * 2",
                    "def get_user(id): return User.query.get(id)",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="DOC001",
                name="문서화되지 않은 공개 API",
                category=RuleCategory.DOCUMENTATION,
                severity=Severity.MEDIUM,
                version="1.0.0",
                description="공개 인터페이스에 대한 문서가 누락됨",
                examples=[
                    "def api_endpoint(request): pass  # 문서 없음",
                    "class PublicService: def method(self): pass  # 문서 없음",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="SCHEMA001",
                name="인덱스 누락",
                category=RuleCategory.SCHEMA,
                severity=Severity.MEDIUM,
                version="1.0.0",
                description="WHERE 절에서 자주 사용되는 컬럼에 인덱스가 없음",
                examples=[
                    "WHERE user_id = ?  # user_id에 인덱스 없음",
                    "ORDER BY created_at DESC  # created_at에 인덱스 없음",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
        ]

        for rule in default_rules:
            self._rules[rule.rule_id] = rule

    def register_rule(self, rule: RuleDefinition) -> None:
        """Register a new rule."""
        if rule.rule_id in self._rules:
            raise ValueError(f"Rule {rule.rule_id} already exists")
        self._rules[rule.rule_id] = rule

    def get_rule(self, rule_id: str) -> RuleDefinition | None:
        """Get rule by ID."""
        return self._rules.get(rule_id)

    def get_rules_by_category(self, category: RuleCategory) -> list[RuleDefinition]:
        """Filter rules by category."""
        return [rule for rule in self._rules.values() if rule.category == category]

    def get_rules_by_severity(self, severity: Severity) -> list[RuleDefinition]:
        """Filter rules by severity."""
        return [rule for rule in self._rules.values() if rule.severity == severity]

    def validate_rule_id(self, rule_id: str) -> bool:
        """Validate if rule ID is valid."""
        return rule_id in self._rules

    def get_all_rules(self) -> dict[str, RuleDefinition]:
        """Get all rules."""
        return self._rules.copy()

    def export_rules(self) -> str:
        """Export rules to JSON format."""
        rules_dict = {rule_id: rule.to_dict() for rule_id, rule in self._rules.items()}
        return json.dumps(rules_dict, indent=2, ensure_ascii=False)

    def import_rules(self, json_str: str) -> None:
        """Import rules from JSON string."""
        rules_dict = json.loads(json_str)
        for rule_id, rule_data in rules_dict.items():
            rule = RuleDefinition.from_dict(rule_data)
            self._rules[rule_id] = rule

    def get_rule_version(self, rule_id: str) -> str | None:
        """Get current version of a rule."""
        rule = self.get_rule(rule_id)
        return rule.version if rule else None

    def is_rule_deprecated(self, rule_id: str) -> bool:
        """Check if rule is deprecated."""
        rule = self.get_rule(rule_id)
        return (
            rule.deprecated if rule else True
        )  # Consider non-existent rules as deprecated


# Global rule registry instance
rule_registry = RuleRegistry()


def get_rule_registry() -> RuleRegistry:
    """Get global rule registry instance."""
    return rule_registry


def validate_evidence_rule_compliance(rule_id: str) -> bool:
    """
    Validate if Evidence rule ID is valid (NFR-5 compliance).

    Args:
        rule_id: Rule ID to validate

    Returns:
        True if rule is valid, False otherwise
    """
    return rule_registry.validate_rule_id(rule_id)


def get_rule_info(rule_id: str) -> dict[str, Any] | None:
    """
    Get detailed information about a rule.

    Args:
        rule_id: Rule ID to look up

    Returns:
        Rule information dictionary or None if not found
    """
    rule = rule_registry.get_rule(rule_id)
    return rule.to_dict() if rule else None
